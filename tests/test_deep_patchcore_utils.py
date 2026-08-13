from pathlib import Path

import numpy as np
from PIL import Image

from qcell.deep_patchcore import binary_auroc, load_mvtec_bottle


def test_binary_auroc_perfect_ranking() -> None:
    labels = np.asarray([0, 0, 1, 1])
    scores = np.asarray([0.1, 0.2, 0.8, 0.9])

    assert binary_auroc(labels, scores) == 1.0


def test_mvtec_loader_assigns_labels_and_masks(tmp_path: Path) -> None:
    bottle = tmp_path / "bottle"
    train = bottle / "train" / "good"
    test_good = bottle / "test" / "good"
    test_broken = bottle / "test" / "broken"
    masks = bottle / "ground_truth" / "broken"
    for directory in (train, test_good, test_broken, masks):
        directory.mkdir(parents=True)

    image = Image.new("RGB", (8, 8), "white")
    image.save(train / "000.png")
    image.save(test_good / "001.png")
    image.save(test_broken / "002.png")
    Image.new("L", (8, 8), 255).save(masks / "002_mask.png")

    train_paths, samples = load_mvtec_bottle(bottle)

    assert len(train_paths) == 1
    assert [sample.label for sample in samples] == [1, 0]
    assert samples[0].mask_path is not None
