from pathlib import Path

from qcell.patch_memory import (
    PatchMemoryDetector,
    classification_metrics,
    generate_normal_training_set,
)
from qcell.vision import generate_demo_pair


def _small_model() -> PatchMemoryDetector:
    return PatchMemoryDetector.fit(generate_normal_training_set(count=8, seed=11))


def test_trained_model_separates_normal_and_scratch() -> None:
    model = _small_model()
    _, normal = generate_demo_pair("normal")
    _, scratch = generate_demo_pair("scratch")

    normal_prediction = model.predict(normal)
    scratch_prediction = model.predict(scratch)

    assert normal_prediction.is_defect is False
    assert scratch_prediction.is_defect is True
    assert scratch_prediction.raw_score > normal_prediction.raw_score


def test_model_round_trip(tmp_path: Path) -> None:
    model = _small_model()
    path = model.save(tmp_path / "model.npz")
    loaded = PatchMemoryDetector.load(path)
    _, scratch = generate_demo_pair("scratch")

    assert loaded.predict(scratch).raw_score == model.predict(scratch).raw_score


def test_classification_metrics() -> None:
    metrics = classification_metrics(
        labels=[0, 0, 1, 1], scores=[0.1, 0.2, 0.8, 0.9], threshold=0.5
    )

    assert metrics["accuracy"] == 1.0
    assert metrics["f1"] == 1.0
    assert metrics["auroc"] == 1.0
