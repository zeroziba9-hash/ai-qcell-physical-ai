from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from qcell.realtime import RealtimeInspectionStore, inspect_frame, records_to_csv


@dataclass
class FakePrediction:
    is_defect: bool
    anomaly_score: float
    raw_score: float
    threshold: float
    latency_ms: float
    overlay: Image.Image


class FakePredictor:
    def __init__(self, is_defect: bool) -> None:
        self.is_defect = is_defect

    def predict(self, image: Image.Image) -> FakePrediction:
        return FakePrediction(self.is_defect, 75.0, 0.6, 0.4, 12.0, image.copy())


def test_inspect_frame_records_and_saves_reject(tmp_path: Path):
    image = Image.fromarray(np.full((80, 120, 3), 180, dtype=np.uint8))
    record, overlay = inspect_frame(image, FakePredictor(True), 12, "camera 1", tmp_path)

    assert record.decision == "REJECT"
    assert Path(record.saved_path).is_file()
    assert overlay.size == image.size
    assert "camera_1" in record.saved_path


def test_store_is_snapshot_safe_and_csv_has_fields():
    image = Image.new("RGB", (32, 32), "white")
    record, overlay = inspect_frame(image, FakePredictor(False), 1, "test")
    store = RealtimeInspectionStore(max_records=2)
    store.update(record, overlay)

    records, latest = store.snapshot()
    csv_text = records_to_csv(records)
    assert records[0].decision == "PASS"
    assert latest is not None
    assert "latency_ms" in csv_text
    assert "PASS" in csv_text
