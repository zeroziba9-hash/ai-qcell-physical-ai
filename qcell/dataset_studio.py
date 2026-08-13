from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
from threading import RLock
from typing import Iterable
from uuid import uuid4

from PIL import Image, ImageOps


LABELS = {"normal", "defect", "unlabeled"}
SPLITS = {"unassigned", "train", "validation", "test", "review"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass(frozen=True)
class DatasetRecord:
    record_id: str
    relative_path: str
    label: str
    defect_type: str
    split: str
    source: str
    created_at: str

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "DatasetRecord":
        return cls(**{field: str(payload.get(field, "")) for field in cls.__annotations__})

    def path(self, root: str | Path) -> Path:
        return Path(root) / self.relative_path

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True)
class TrainingBundle:
    train_normal: tuple[DatasetRecord, ...]
    validation_normal: tuple[DatasetRecord, ...]
    calibration: tuple[DatasetRecord, ...]
    evaluation: tuple[DatasetRecord, ...]


class DatasetStudio:
    """Persistent, manifest-backed image dataset used by the active-learning UI."""

    schema_version = 1

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.images_dir = self.root / "images"
        self.manifest_path = self.root / "manifest.json"
        self._lock = RLock()

    def _read_payload(self) -> dict[str, object]:
        if not self.manifest_path.exists():
            return {"schema_version": self.schema_version, "updated_at": utc_now(), "records": []}
        payload = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        if int(payload.get("schema_version", 0)) != self.schema_version:
            raise ValueError("unsupported dataset manifest schema")
        return payload

    def _write_records(self, records: Iterable[DatasetRecord]) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": self.schema_version,
            "updated_at": utc_now(),
            "records": [record.to_dict() for record in records],
        }
        temporary = self.manifest_path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        temporary.replace(self.manifest_path)

    def records(
        self,
        label: str | None = None,
        split: str | None = None,
    ) -> list[DatasetRecord]:
        payload = self._read_payload()
        records = [DatasetRecord.from_dict(item) for item in payload.get("records", [])]
        if label is not None:
            records = [record for record in records if record.label == label]
        if split is not None:
            records = [record for record in records if record.split == split]
        return sorted(records, key=lambda record: (record.created_at, record.record_id))

    def add_image(
        self,
        image: Image.Image,
        label: str,
        defect_type: str = "",
        source: str = "manual",
        split: str = "unassigned",
    ) -> DatasetRecord:
        _validate_label_split(label, split)
        if label != "defect":
            defect_type = ""
        record_id = uuid4().hex[:16]
        relative_path = Path("images") / f"{record_id}.png"
        destination = self.root / relative_path
        with self._lock:
            destination.parent.mkdir(parents=True, exist_ok=True)
            prepared = ImageOps.exif_transpose(image).convert("RGB")
            prepared.save(destination, format="PNG", optimize=True)
            record = DatasetRecord(
                record_id=record_id,
                relative_path=relative_path.as_posix(),
                label=label,
                defect_type=defect_type.strip(),
                split=split,
                source=source.strip() or "manual",
                created_at=utc_now(),
            )
            records = self.records()
            records.append(record)
            self._write_records(records)
        return record

    def add_path(
        self,
        path: str | Path,
        label: str,
        defect_type: str = "",
        source: str | None = None,
        split: str = "unassigned",
    ) -> DatasetRecord:
        source_path = Path(path)
        with Image.open(source_path) as image:
            return self.add_image(
                image,
                label=label,
                defect_type=defect_type,
                source=source or str(source_path),
                split=split,
            )

    def update_record(
        self,
        record_id: str,
        *,
        label: str | None = None,
        defect_type: str | None = None,
        split: str | None = None,
    ) -> DatasetRecord:
        with self._lock:
            records = self.records()
            for index, record in enumerate(records):
                if record.record_id != record_id:
                    continue
                next_label = label or record.label
                next_split = split or record.split
                _validate_label_split(next_label, next_split)
                next_defect = record.defect_type if defect_type is None else defect_type.strip()
                if next_label != "defect":
                    next_defect = ""
                updated = replace(
                    record,
                    label=next_label,
                    defect_type=next_defect,
                    split=next_split,
                )
                records[index] = updated
                self._write_records(records)
                return updated
        raise KeyError(f"dataset record not found: {record_id}")

    def assign_splits(
        self,
        train_ratio: float = 0.8,
        validation_ratio: float = 0.1,
        seed: int = 42,
    ) -> list[DatasetRecord]:
        with self._lock:
            records = assign_record_splits(
                self.records(),
                train_ratio=train_ratio,
                validation_ratio=validation_ratio,
                seed=seed,
            )
            self._write_records(records)
        return records

    def training_bundle(self) -> TrainingBundle:
        records = self.records()
        train_normal = tuple(
            record for record in records if record.label == "normal" and record.split == "train"
        )
        validation_normal = tuple(
            record
            for record in records
            if record.label == "normal" and record.split == "validation"
        )
        calibration = tuple(
            record
            for record in records
            if record.label in {"normal", "defect"} and record.split == "validation"
        )
        evaluation = tuple(
            record
            for record in records
            if record.label in {"normal", "defect"} and record.split == "test"
        )
        return TrainingBundle(train_normal, validation_normal, calibration, evaluation)

    def statistics(self) -> dict[str, object]:
        records = self.records()
        by_label = {label: sum(record.label == label for record in records) for label in LABELS}
        by_split = {split: sum(record.split == split for record in records) for split in SPLITS}
        return {"total": len(records), "by_label": by_label, "by_split": by_split}

    def fingerprint(self) -> str:
        canonical = json.dumps(
            [record.to_dict() for record in self.records()],
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]

    def seed_from_mvtec(
        self,
        mvtec_root: str | Path,
        normal_count: int = 40,
        defect_count: int = 12,
    ) -> int:
        root = Path(mvtec_root)
        normal_paths = sorted((root / "train" / "good").glob("*.png"))[:normal_count]
        defect_paths = []
        for directory in sorted((root / "test").iterdir()) if (root / "test").exists() else []:
            if directory.is_dir() and directory.name != "good":
                defect_paths.extend(sorted(directory.glob("*.png")))
        defect_paths = defect_paths[:defect_count]
        existing_sources = {record.source for record in self.records()}
        added = 0
        for path in normal_paths:
            source = f"mvtec:{path.relative_to(root).as_posix()}"
            if source not in existing_sources:
                self.add_path(path, "normal", source=source)
                existing_sources.add(source)
                added += 1
        for path in defect_paths:
            source = f"mvtec:{path.relative_to(root).as_posix()}"
            if source not in existing_sources:
                self.add_path(path, "defect", defect_type=path.parent.name, source=source)
                existing_sources.add(source)
                added += 1
        if added:
            self.assign_splits(train_ratio=0.8, validation_ratio=0.1, seed=42)
        return added


def assign_record_splits(
    records: list[DatasetRecord],
    train_ratio: float = 0.8,
    validation_ratio: float = 0.1,
    seed: int = 42,
) -> list[DatasetRecord]:
    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be between 0 and 1")
    if not 0 <= validation_ratio < 1 or train_ratio + validation_ratio >= 1:
        raise ValueError("validation_ratio must keep a positive test split")
    rng = random.Random(seed)
    updated = {record.record_id: record for record in records}

    normal = [record for record in records if record.label == "normal"]
    rng.shuffle(normal)
    normal_count = len(normal)
    if normal_count >= 3:
        validation_count = max(1, round(normal_count * validation_ratio))
        test_count = max(1, normal_count - round(normal_count * train_ratio) - validation_count)
        train_count = normal_count - validation_count - test_count
        if train_count < 1:
            train_count, validation_count, test_count = normal_count - 2, 1, 1
    else:
        train_count, validation_count = normal_count, 0
        test_count = 0
    for index, record in enumerate(normal):
        split = "train" if index < train_count else "validation"
        if index >= train_count + validation_count:
            split = "test"
        updated[record.record_id] = replace(record, split=split)

    defects = [record for record in records if record.label == "defect"]
    rng.shuffle(defects)
    validation_defects = max(1, len(defects) // 2) if len(defects) > 1 else 0
    for index, record in enumerate(defects):
        split = "validation" if index < validation_defects else "test"
        updated[record.record_id] = replace(record, split=split)

    for record in records:
        if record.label == "unlabeled":
            updated[record.record_id] = replace(record, split="review")
    return [updated[record.record_id] for record in records]


def _validate_label_split(label: str, split: str) -> None:
    if label not in LABELS:
        raise ValueError(f"unsupported label: {label}")
    if split not in SPLITS:
        raise ValueError(f"unsupported split: {split}")
