from pathlib import Path

import numpy as np
from PIL import Image

import qcell.active_learning as active_learning
from qcell.active_learning import TrainingConfig, calibrate_threshold, curve_points
from qcell.dataset_studio import DatasetStudio
from qcell.model_registry import ModelRegistry


class FakeDeepPatchCore:
    def __init__(self, device=None) -> None:
        self.device = device or "cpu"
        self.threshold = 0.3

    def fit(self, train_paths, validation_paths, **kwargs):
        assert len(train_paths) >= 3
        assert validation_paths
        return {"threshold": self.threshold, "training_seconds": 0.01, "device": self.device}

    def score_paths(self, paths, batch_size=8):
        return [float(np.asarray(Image.open(path)).mean() / 255.0) for path in paths]

    def save(self, path, metadata=None):
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"fake-model")
        return destination


def test_threshold_calibration_separates_labels() -> None:
    result = calibrate_threshold([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])
    assert 0.2 < result["threshold"] < 0.8
    assert result["f1"] == 1.0
    assert result["strategy"] == "max-f1-balanced-tiebreak"
    curves = curve_points([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])
    assert curves["roc"][0] == {"fpr": 0.0, "tpr": 0.0}


def test_threshold_normal_only_uses_fallback() -> None:
    result = calibrate_threshold([0, 0], [0.1, 0.2], fallback_threshold=0.25)
    assert result["threshold"] == 0.25
    assert result["strategy"] == "normal-only-fallback"


def test_training_registers_version_with_fake_model(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(active_learning, "DeepPatchCore", FakeDeepPatchCore)
    dataset = DatasetStudio(tmp_path / "dataset")
    for index in range(10):
        dataset.add_image(Image.new("RGB", (16, 16), (20 + index,) * 3), "normal")
    for index in range(4):
        dataset.add_image(Image.new("RGB", (16, 16), (220 + index,) * 3), "defect", "scratch")
    dataset.assign_splits(0.8, 0.1, seed=42)
    registry = ModelRegistry(tmp_path / "registry")

    result = active_learning.train_and_register(
        dataset,
        registry,
        TrainingConfig(batch_size=2, candidate_size=50, coreset_size=10),
        "Fake Production Candidate",
    )
    assert result.metrics["f1"] == 1.0
    assert registry.get(result.version.version_id).display_name == "Fake Production Candidate"
    assert (registry.root / result.version.model_path).is_file()
