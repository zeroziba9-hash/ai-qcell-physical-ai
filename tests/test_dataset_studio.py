from pathlib import Path

from PIL import Image

from qcell.dataset_studio import DatasetStudio


def sample_image(value: int = 80) -> Image.Image:
    return Image.new("RGB", (32, 32), (value, value, value))


def test_dataset_add_update_and_statistics(tmp_path: Path) -> None:
    studio = DatasetStudio(tmp_path / "dataset")
    normal = studio.add_image(sample_image(), "normal", source="camera")
    defect = studio.add_image(sample_image(200), "defect", "scratch", source="upload")
    unlabeled = studio.add_image(sample_image(120), "unlabeled")

    stats = studio.statistics()
    assert stats["total"] == 3
    assert stats["by_label"] == {"normal": 1, "defect": 1, "unlabeled": 1}
    assert normal.path(studio.root).is_file()
    assert defect.defect_type == "scratch"

    updated = studio.update_record(unlabeled.record_id, label="defect", defect_type="dent")
    assert updated.label == "defect"
    assert updated.defect_type == "dent"
    assert len(studio.fingerprint()) == 16


def test_dataset_split_keeps_defects_out_of_training(tmp_path: Path) -> None:
    studio = DatasetStudio(tmp_path / "dataset")
    for index in range(10):
        studio.add_image(sample_image(index), "normal", source=f"normal-{index}")
    for index in range(4):
        studio.add_image(sample_image(200 + index), "defect", "scratch")
    studio.add_image(sample_image(100), "unlabeled")

    studio.assign_splits(train_ratio=0.8, validation_ratio=0.1, seed=7)
    bundle = studio.training_bundle()
    assert len(bundle.train_normal) == 8
    assert len(bundle.validation_normal) == 1
    assert len(bundle.calibration) == 3
    assert len(bundle.evaluation) == 3
    assert all(record.label == "normal" for record in bundle.train_normal)
    assert studio.statistics()["by_split"]["review"] == 1


def test_mvtec_seed_is_idempotent(tmp_path: Path) -> None:
    mvtec = tmp_path / "mvtec"
    for index in range(3):
        path = mvtec / "train" / "good" / f"{index:03d}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        sample_image(index).save(path)
    for index in range(2):
        path = mvtec / "test" / "scratch" / f"{index:03d}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        sample_image(200 + index).save(path)

    studio = DatasetStudio(tmp_path / "dataset")
    assert studio.seed_from_mvtec(mvtec, normal_count=3, defect_count=2) == 5
    assert studio.seed_from_mvtec(mvtec, normal_count=3, defect_count=2) == 0
    assert studio.statistics()["total"] == 5
