from pathlib import Path

from PIL import Image
import pytest

from qcell.dataset_studio import DatasetStudio
from qcell.review_queue import ReviewQueue, is_uncertain


def test_review_resolution_feeds_dataset(tmp_path: Path) -> None:
    queue = ReviewQueue(tmp_path / "review")
    dataset = DatasetStudio(tmp_path / "dataset")
    case = queue.add_case(
        Image.new("RGB", (32, 32), "white"),
        predicted_label="defect",
        raw_score=0.51,
        threshold=0.5,
        source="camera",
        model_version="v1",
    )
    assert case.uncertainty_percent == 98.0
    assert len(queue.cases("pending")) == 1

    resolved, record = queue.resolve(case.case_id, "normal", dataset)
    assert resolved.status == "corrected"
    assert record.label == "normal"
    assert dataset.statistics()["total"] == 1
    with pytest.raises(ValueError):
        queue.resolve(case.case_id, "normal", dataset)


def test_uncertainty_margin() -> None:
    assert is_uncertain(0.55, 0.5, 0.15)
    assert not is_uncertain(0.8, 0.5, 0.15)
    assert is_uncertain(0.2, 0.0)
