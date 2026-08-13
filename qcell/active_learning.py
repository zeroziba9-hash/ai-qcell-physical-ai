from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from PIL import Image

from .dataset_studio import DatasetRecord, DatasetStudio
from .deep_patchcore import DeepPatchCore
from .model_registry import ModelRegistry, ModelVersion, new_version_id
from .patch_memory import classification_metrics


@dataclass(frozen=True)
class TrainingConfig:
    batch_size: int = 8
    candidate_size: int = 4000
    coreset_size: int = 256
    seed: int = 42
    device: str | None = None


@dataclass(frozen=True)
class TrainingResult:
    version: ModelVersion
    metadata: dict[str, object]
    metrics: dict[str, object]
    calibration: dict[str, object]
    score_rows: tuple[dict[str, object], ...]


def calibrate_threshold(
    labels: list[int] | np.ndarray,
    scores: list[float] | np.ndarray,
    fallback_threshold: float | None = None,
) -> dict[str, object]:
    truth = np.asarray(labels, dtype=np.int32).reshape(-1)
    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    if len(truth) != len(values) or not len(truth):
        raise ValueError("labels and scores must be non-empty and have equal length")
    if not np.isfinite(values).all():
        raise ValueError("scores must be finite")
    unique_labels = set(int(value) for value in np.unique(truth))
    if not unique_labels.issubset({0, 1}):
        raise ValueError("labels must be binary")

    if unique_labels != {0, 1}:
        if fallback_threshold is None:
            fallback_threshold = float(np.max(values) * 1.05 + 1e-8)
        return {
            "threshold": float(fallback_threshold),
            "strategy": "normal-only-fallback",
            "f1": 0.0,
            "balanced_accuracy": 0.0,
            "candidate_count": 1,
        }

    unique_scores = np.unique(values)
    epsilon = max(float(np.ptp(unique_scores)) * 1e-6, 1e-9)
    candidates = [float(unique_scores[0] - epsilon)]
    candidates.extend(
        float((left + right) / 2.0)
        for left, right in zip(unique_scores[:-1], unique_scores[1:])
    )
    candidates.append(float(unique_scores[-1] + epsilon))

    best: tuple[tuple[float, float, float, float], float, dict[str, float]] | None = None
    for threshold in candidates:
        predicted = values > threshold
        tp = int(((truth == 1) & predicted).sum())
        tn = int(((truth == 0) & ~predicted).sum())
        fp = int(((truth == 0) & predicted).sum())
        fn = int(((truth == 1) & ~predicted).sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        specificity = tn / (tn + fp) if tn + fp else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        balanced = (recall + specificity) / 2.0
        key = (f1, balanced, specificity, -threshold)
        metrics = {
            "f1": f1,
            "balanced_accuracy": balanced,
            "precision": precision,
            "recall": recall,
            "specificity": specificity,
        }
        if best is None or key > best[0]:
            best = (key, threshold, metrics)

    assert best is not None
    return {
        "threshold": round(best[1], 8),
        "strategy": "max-f1-balanced-tiebreak",
        "f1": round(best[2]["f1"], 4),
        "balanced_accuracy": round(best[2]["balanced_accuracy"], 4),
        "precision": round(best[2]["precision"], 4),
        "recall": round(best[2]["recall"], 4),
        "specificity": round(best[2]["specificity"], 4),
        "candidate_count": len(candidates),
    }


def curve_points(
    labels: list[int] | np.ndarray,
    scores: list[float] | np.ndarray,
) -> dict[str, list[dict[str, float]]]:
    truth = np.asarray(labels, dtype=np.int32).reshape(-1)
    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    if len(truth) != len(values):
        raise ValueError("labels and scores must have equal length")
    thresholds = [float("inf"), *sorted((float(value) for value in np.unique(values)), reverse=True)]
    positives = max(int((truth == 1).sum()), 1)
    negatives = max(int((truth == 0).sum()), 1)
    roc: list[dict[str, float]] = []
    precision_recall: list[dict[str, float]] = []
    for threshold in thresholds:
        predicted = values > threshold
        tp = int(((truth == 1) & predicted).sum())
        fp = int(((truth == 0) & predicted).sum())
        tpr = tp / positives
        fpr = fp / negatives
        precision = tp / (tp + fp) if tp + fp else 1.0
        roc.append({"fpr": round(fpr, 6), "tpr": round(tpr, 6)})
        precision_recall.append(
            {"recall": round(tpr, 6), "precision": round(precision, 6)}
        )
    return {"roc": roc, "precision_recall": precision_recall}


def train_and_register(
    dataset: DatasetStudio,
    registry: ModelRegistry,
    config: TrainingConfig,
    display_name: str,
) -> TrainingResult:
    bundle = dataset.training_bundle()
    if len(bundle.train_normal) < 3:
        raise ValueError("at least three normal training images are required")
    if not bundle.validation_normal:
        raise ValueError("at least one normal validation image is required")

    started = perf_counter()
    model = DeepPatchCore(device=config.device)
    metadata = model.fit(
        [record.path(dataset.root) for record in bundle.train_normal],
        [record.path(dataset.root) for record in bundle.validation_normal],
        batch_size=config.batch_size,
        candidate_size=config.candidate_size,
        coreset_size=config.coreset_size,
        seed=config.seed,
    )

    calibration_records = list(bundle.calibration)
    calibration_scores = model.score_paths(
        [record.path(dataset.root) for record in calibration_records],
        batch_size=config.batch_size,
    )
    calibration_labels = [1 if record.label == "defect" else 0 for record in calibration_records]
    calibration = calibrate_threshold(
        calibration_labels,
        calibration_scores,
        fallback_threshold=model.threshold,
    )
    model.threshold = float(calibration["threshold"])

    evaluation_records = list(bundle.evaluation) or calibration_records
    evaluation_scores = model.score_paths(
        [record.path(dataset.root) for record in evaluation_records],
        batch_size=config.batch_size,
    )
    evaluation_labels = [1 if record.label == "defect" else 0 for record in evaluation_records]
    metrics = classification_metrics(evaluation_labels, evaluation_scores, model.threshold)
    if len(set(evaluation_labels)) < 2:
        metrics["auroc"] = 0.0
    metrics["curves"] = curve_points(evaluation_labels, evaluation_scores)
    metrics["calibration"] = calibration

    score_rows = tuple(
        {
            "record_id": record.record_id,
            "label": record.label,
            "defect_type": record.defect_type,
            "split": record.split,
            "score": round(float(score), 6),
            "prediction": "defect" if score > model.threshold else "normal",
        }
        for record, score in zip(evaluation_records, evaluation_scores)
    )

    version_id = new_version_id("custom")
    training_seconds = perf_counter() - started
    metadata.update(
        {
            "dataset": "AI-QCell Dataset Studio",
            "dataset_fingerprint": dataset.fingerprint(),
            "dataset_statistics": dataset.statistics(),
            "training_config": asdict(config),
            "calibration": calibration,
            "evaluation_samples": len(evaluation_records),
        }
    )
    staging_dir = registry.root / ".staging"
    staging_dir.mkdir(parents=True, exist_ok=True)
    staging_path = staging_dir / f"{version_id}.pt"
    model.save(staging_path, metadata)
    try:
        version = registry.register(
            staging_path,
            version_id=version_id,
            display_name=display_name,
            metadata=metadata,
            metrics=metrics,
            dataset_fingerprint=dataset.fingerprint(),
            threshold=model.threshold,
            training_seconds=training_seconds,
        )
    finally:
        staging_path.unlink(missing_ok=True)
    return TrainingResult(version, metadata, metrics, calibration, score_rows)


def ensure_baseline_registered(
    registry: ModelRegistry,
    model_path: str | Path,
    metadata_path: str | Path | None = None,
    report_path: str | Path | None = None,
) -> ModelVersion:
    version_id = "mvtec-bottle-v1"
    try:
        return registry.get(version_id)
    except KeyError:
        pass
    metadata = _read_json(metadata_path)
    report = _read_json(report_path)
    metrics = dict(report.get("image_metrics", {}))
    threshold = float(metadata.get("threshold", report.get("threshold", 0.0)))
    version = registry.register(
        model_path,
        version_id=version_id,
        display_name="MVTec Bottle Baseline",
        metadata=metadata,
        metrics=metrics,
        dataset_fingerprint="mvtec-bottle-official",
        threshold=threshold,
        training_seconds=float(metadata.get("training_seconds", 0.0)),
    )
    if registry.deployed() is None:
        registry.deploy(version.version_id, reason="initial-baseline")
    return version


def load_registry_model(
    registry: ModelRegistry,
    fallback_model: str | Path,
    device: str | None = None,
) -> tuple[DeepPatchCore, str, Path]:
    model_path, version_id = registry.resolve_model_path(fallback_model)
    return DeepPatchCore.load(model_path, device=device), version_id, model_path


def evaluate_image_for_review(
    image: Image.Image,
    model: DeepPatchCore,
) -> dict[str, object]:
    prediction = model.predict(image.convert("RGB"))
    return {
        "predicted_label": "defect" if prediction.is_defect else "normal",
        "raw_score": prediction.raw_score,
        "threshold": prediction.threshold,
        "anomaly_score": prediction.anomaly_score,
        "latency_ms": prediction.latency_ms,
        "overlay": prediction.overlay,
    }


def _read_json(path: str | Path | None) -> dict[str, object]:
    if path is None or not Path(path).is_file():
        return {}
    return dict(json.loads(Path(path).read_text(encoding="utf-8")))
