from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import json
from pathlib import Path
from threading import RLock
from uuid import uuid4

from PIL import Image, ImageOps

from .dataset_studio import DatasetRecord, DatasetStudio


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass(frozen=True)
class ReviewCase:
    case_id: str
    relative_path: str
    predicted_label: str
    raw_score: float
    threshold: float
    source: str
    model_version: str
    status: str
    actual_label: str
    defect_type: str
    dataset_record_id: str
    created_at: str
    resolved_at: str

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "ReviewCase":
        return cls(
            case_id=str(payload.get("case_id", "")),
            relative_path=str(payload.get("relative_path", "")),
            predicted_label=str(payload.get("predicted_label", "")),
            raw_score=float(payload.get("raw_score", 0.0)),
            threshold=float(payload.get("threshold", 0.0)),
            source=str(payload.get("source", "")),
            model_version=str(payload.get("model_version", "")),
            status=str(payload.get("status", "pending")),
            actual_label=str(payload.get("actual_label", "")),
            defect_type=str(payload.get("defect_type", "")),
            dataset_record_id=str(payload.get("dataset_record_id", "")),
            created_at=str(payload.get("created_at", "")),
            resolved_at=str(payload.get("resolved_at", "")),
        )

    def path(self, root: str | Path) -> Path:
        return Path(root) / self.relative_path

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @property
    def uncertainty_percent(self) -> float:
        if self.threshold <= 0:
            return 100.0
        distance = abs(self.raw_score - self.threshold) / self.threshold
        return round(max(0.0, 100.0 * (1.0 - min(distance, 1.0))), 1)


class ReviewQueue:
    schema_version = 1

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.images_dir = self.root / "images"
        self.manifest_path = self.root / "queue.json"
        self._lock = RLock()

    def _read(self) -> list[ReviewCase]:
        if not self.manifest_path.exists():
            return []
        payload = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        if int(payload.get("schema_version", 0)) != self.schema_version:
            raise ValueError("unsupported review queue schema")
        return [ReviewCase.from_dict(item) for item in payload.get("cases", [])]

    def _write(self, cases: list[ReviewCase]) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": self.schema_version,
            "updated_at": _utc_now(),
            "cases": [case.to_dict() for case in cases],
        }
        temporary = self.manifest_path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        temporary.replace(self.manifest_path)

    def cases(self, status: str | None = None) -> list[ReviewCase]:
        cases = self._read()
        if status is not None:
            cases = [case for case in cases if case.status == status]
        return sorted(cases, key=lambda case: (case.created_at, case.case_id), reverse=True)

    def add_case(
        self,
        image: Image.Image,
        predicted_label: str,
        raw_score: float,
        threshold: float,
        source: str,
        model_version: str,
    ) -> ReviewCase:
        if predicted_label not in {"normal", "defect"}:
            raise ValueError("predicted_label must be normal or defect")
        case_id = uuid4().hex[:16]
        relative_path = Path("images") / f"{case_id}.png"
        destination = self.root / relative_path
        with self._lock:
            destination.parent.mkdir(parents=True, exist_ok=True)
            ImageOps.exif_transpose(image).convert("RGB").save(
                destination, format="PNG", optimize=True
            )
            case = ReviewCase(
                case_id=case_id,
                relative_path=relative_path.as_posix(),
                predicted_label=predicted_label,
                raw_score=float(raw_score),
                threshold=float(threshold),
                source=source,
                model_version=model_version,
                status="pending",
                actual_label="",
                defect_type="",
                dataset_record_id="",
                created_at=_utc_now(),
                resolved_at="",
            )
            cases = self._read()
            cases.append(case)
            self._write(cases)
        return case

    def import_realtime_rejects(self, directory: str | Path) -> int:
        folder = Path(directory)
        if not folder.exists():
            return 0
        existing = {case.source for case in self.cases()}
        added = 0
        for path in sorted(folder.iterdir()):
            if path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
                continue
            source = f"realtime:{path.resolve()}"
            if source in existing:
                continue
            with Image.open(path) as image:
                self.add_case(
                    image,
                    predicted_label="defect",
                    raw_score=0.0,
                    threshold=0.0,
                    source=source,
                    model_version="realtime-import",
                )
            existing.add(source)
            added += 1
        return added

    def resolve(
        self,
        case_id: str,
        actual_label: str,
        dataset: DatasetStudio,
        defect_type: str = "",
    ) -> tuple[ReviewCase, DatasetRecord]:
        if actual_label not in {"normal", "defect"}:
            raise ValueError("actual_label must be normal or defect")
        with self._lock:
            cases = self._read()
            for index, case in enumerate(cases):
                if case.case_id != case_id:
                    continue
                if case.status != "pending":
                    raise ValueError("review case has already been resolved")
                image_path = case.path(self.root)
                dataset_record = dataset.add_path(
                    image_path,
                    label=actual_label,
                    defect_type=defect_type,
                    source=f"review:{case.case_id}",
                )
                status = "confirmed" if actual_label == case.predicted_label else "corrected"
                resolved = replace(
                    case,
                    status=status,
                    actual_label=actual_label,
                    defect_type=defect_type if actual_label == "defect" else "",
                    dataset_record_id=dataset_record.record_id,
                    resolved_at=_utc_now(),
                )
                cases[index] = resolved
                self._write(cases)
                return resolved, dataset_record
        raise KeyError(f"review case not found: {case_id}")


def is_uncertain(raw_score: float, threshold: float, margin: float = 0.15) -> bool:
    if threshold <= 0:
        return True
    if not 0 <= margin <= 1:
        raise ValueError("margin must be between 0 and 1")
    return abs(float(raw_score) - float(threshold)) / float(threshold) <= margin
