from pathlib import Path

import numpy as np
from PIL import Image
import torch

from qcell.edge_runtime import (
    EdgePrediction,
    benchmark_predictor,
    image_to_input,
    make_edge_prediction,
    nearest_patch_distances,
)


class FakePredictor:
    def predict(self, image: Image.Image) -> EdgePrediction:
        score = float(np.asarray(image).mean() / 255.0)
        return make_edge_prediction(
            image.resize((224, 224)),
            score,
            np.full((28, 28), score, dtype=np.float32),
            threshold=0.5,
            latency_ms=1.0,
            backend="fake",
        )


def test_nearest_patch_distance_matches_cdist() -> None:
    generator = torch.Generator().manual_seed(7)
    embeddings = torch.randn(2, 5, 4, generator=generator)
    memory = torch.randn(6, 4, generator=generator)
    expected = torch.cdist(embeddings, memory).amin(dim=-1)
    actual = nearest_patch_distances(embeddings, memory)
    assert torch.allclose(actual, expected, atol=1e-5)


def test_image_input_shape_and_range() -> None:
    array = image_to_input(Image.new("RGB", (40, 80), (255, 128, 0)))
    assert array.shape == (1, 3, 224, 224)
    assert array.dtype == np.float32
    assert 0 <= float(array.min()) <= float(array.max()) <= 1


def test_edge_prediction_and_benchmark() -> None:
    images = [Image.new("RGB", (32, 32), (30,) * 3), Image.new("RGB", (32, 32), (230,) * 3)]
    predictor = FakePredictor()
    references = [predictor.predict(image) for image in images]
    report = benchmark_predictor(
        "Fake",
        "CPU",
        predictor,
        images,
        reference_scores=[prediction.raw_score for prediction in references],
        reference_decisions=[prediction.is_defect for prediction in references],
        warmup_runs=1,
        measured_runs=3,
    )
    assert report.sample_count == 2
    assert report.decision_agreement == 1.0
    assert report.score_max_error == 0.0
    assert report.fps > 0
    assert references[0].is_defect is False
    assert references[1].is_defect is True
