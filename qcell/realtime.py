from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path
from threading import RLock
from time import perf_counter
from typing import Callable, Protocol

import av
import cv2
import imageio.v2 as imageio
import numpy as np
from PIL import Image
from streamlit_webrtc import VideoProcessorBase


class PredictionLike(Protocol):
    is_defect: bool
    anomaly_score: float
    raw_score: float
    threshold: float
    latency_ms: float
    overlay: Image.Image


class Predictor(Protocol):
    def predict(self, image: Image.Image) -> PredictionLike: ...


@dataclass(frozen=True)
class RealtimeRecord:
    timestamp: str
    frame_index: int
    source: str
    decision: str
    anomaly_score: float
    raw_score: float
    threshold: float
    latency_ms: float
    saved_path: str = ""

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class VideoAnalysisSummary:
    output_path: Path
    records: tuple[RealtimeRecord, ...]
    frames_read: int
    frames_analyzed: int
    source_fps: float
    elapsed_seconds: float

    @property
    def rejects(self) -> int:
        return sum(record.decision == "REJECT" for record in self.records)


class RealtimeInspectionStore:
    """Thread-safe bridge between a WebRTC worker and the Streamlit UI."""

    def __init__(self, max_records: int = 300) -> None:
        self._lock = RLock()
        self._records: deque[RealtimeRecord] = deque(maxlen=max_records)
        self._latest_overlay: Image.Image | None = None

    def update(self, record: RealtimeRecord, overlay: Image.Image) -> None:
        with self._lock:
            self._records.append(record)
            self._latest_overlay = overlay.copy()

    def snapshot(self) -> tuple[list[RealtimeRecord], Image.Image | None]:
        with self._lock:
            overlay = self._latest_overlay.copy() if self._latest_overlay else None
            return list(self._records), overlay

    def clear(self) -> None:
        with self._lock:
            self._records.clear()
            self._latest_overlay = None


def _timestamp() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def render_status_frame(image: Image.Image, prediction: PredictionLike) -> Image.Image:
    canvas = prediction.overlay.convert("RGB").copy()
    array = np.asarray(canvas).copy()
    decision = "REJECT" if prediction.is_defect else "PASS"
    color = (38, 48, 238) if prediction.is_defect else (72, 190, 68)
    cv2.rectangle(array, (0, 0), (array.shape[1], 34), (10, 18, 32), -1)
    cv2.putText(
        array,
        f"{decision}  SCORE {prediction.raw_score:.4f}  {prediction.latency_ms:.0f} ms",
        (8, 23),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        color,
        1,
        cv2.LINE_AA,
    )
    return Image.fromarray(array)


def save_defect_frame(
    image: Image.Image,
    output_dir: str | Path,
    source: str,
    frame_index: int,
) -> Path:
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    safe_source = "".join(character if character.isalnum() else "_" for character in source)
    path = directory / f"{safe_source}_{frame_index:06d}.jpg"
    image.convert("RGB").save(path, quality=92)
    return path


def inspect_frame(
    image: Image.Image,
    predictor: Predictor,
    frame_index: int,
    source: str,
    defect_dir: str | Path | None = None,
) -> tuple[RealtimeRecord, Image.Image]:
    prediction = predictor.predict(image.convert("RGB"))
    overlay = render_status_frame(image, prediction)
    saved_path = ""
    if prediction.is_defect and defect_dir is not None:
        saved_path = str(save_defect_frame(overlay, defect_dir, source, frame_index))
    record = RealtimeRecord(
        timestamp=_timestamp(),
        frame_index=frame_index,
        source=source,
        decision="REJECT" if prediction.is_defect else "PASS",
        anomaly_score=float(prediction.anomaly_score),
        raw_score=float(prediction.raw_score),
        threshold=float(prediction.threshold),
        latency_ms=float(prediction.latency_ms),
        saved_path=saved_path,
    )
    return record, overlay


class RealtimeVideoProcessor(VideoProcessorBase):
    def __init__(
        self,
        predictor: Predictor,
        store: RealtimeInspectionStore,
        inspect_every: int = 12,
        defect_dir: str | Path | None = None,
    ) -> None:
        self.predictor = predictor
        self.store = store
        self.inspect_every = max(1, int(inspect_every))
        self.defect_dir = defect_dir
        self.frame_index = 0
        self._predict_lock = RLock()
        self._last_overlay: Image.Image | None = None

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        frame_array = frame.to_ndarray(format="rgb24")
        image = Image.fromarray(frame_array)
        self.frame_index += 1
        if self.frame_index == 1 or self.frame_index % self.inspect_every == 0:
            with self._predict_lock:
                record, overlay = inspect_frame(
                    image,
                    self.predictor,
                    self.frame_index,
                    "webcam",
                    self.defect_dir,
                )
            self._last_overlay = overlay
            self.store.update(record, overlay)
        output = self._last_overlay or image
        resized = output.resize(image.size, Image.Resampling.BILINEAR)
        return av.VideoFrame.from_ndarray(np.asarray(resized), format="rgb24")


def analyze_video(
    video_path: str | Path,
    predictor: Predictor,
    output_path: str | Path,
    inspect_every: int = 15,
    max_frames: int = 900,
    defect_dir: str | Path | None = None,
    on_record: Callable[[RealtimeRecord, Image.Image], None] | None = None,
) -> VideoAnalysisSummary:
    source = Path(video_path)
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise ValueError(f"cannot open video: {source}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if not np.isfinite(fps) or fps <= 0:
        fps = 20.0
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(str(output), fps=fps, codec="libx264", quality=7)
    started = perf_counter()
    records: list[RealtimeRecord] = []
    latest_overlay: Image.Image | None = None
    frame_index = 0
    inspect_every = max(1, int(inspect_every))
    try:
        while frame_index < max_frames:
            ok, bgr = capture.read()
            if not ok:
                break
            frame_index += 1
            image = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            if frame_index == 1 or frame_index % inspect_every == 0:
                record, latest_overlay = inspect_frame(
                    image,
                    predictor,
                    frame_index,
                    source.stem,
                    defect_dir,
                )
                records.append(record)
                if on_record:
                    on_record(record, latest_overlay)
            rendered = (latest_overlay or image).resize(image.size, Image.Resampling.BILINEAR)
            writer.append_data(np.asarray(rendered))
    finally:
        capture.release()
        writer.close()
    return VideoAnalysisSummary(
        output_path=output,
        records=tuple(records),
        frames_read=frame_index,
        frames_analyzed=len(records),
        source_fps=fps,
        elapsed_seconds=round(perf_counter() - started, 3),
    )


def records_to_csv(records: list[RealtimeRecord] | tuple[RealtimeRecord, ...]) -> str:
    if not records:
        return "timestamp,frame_index,source,decision,anomaly_score,raw_score,threshold,latency_ms,saved_path\n"
    import csv

    buffer = StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(records[0].to_dict()))
    writer.writeheader()
    writer.writerows(record.to_dict() for record in records)
    return buffer.getvalue()
