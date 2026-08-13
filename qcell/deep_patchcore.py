from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
from PIL import Image, ImageDraw
import torch
from torch import Tensor
import torch.nn.functional as functional
from torchvision.models import ResNet18_Weights, resnet18
from torchvision.models.feature_extraction import create_feature_extractor


@dataclass(frozen=True)
class MVTecSample:
    path: Path
    defect_type: str
    label: int
    mask_path: Path | None


@dataclass(frozen=True)
class DeepPatchPrediction:
    is_defect: bool
    anomaly_score: float
    raw_score: float
    threshold: float
    latency_ms: float
    anomaly_map: np.ndarray
    overlay: Image.Image
    prepared_image: Image.Image


class DeepPatchCore:
    """PatchCore-style detector using ImageNet ResNet18 layer2/layer3 features."""

    image_size = (224, 224)
    grid_size = 28
    feature_dimensions = 384

    def __init__(
        self,
        memory_bank: Tensor | None = None,
        threshold: float = 0.0,
        device: str | None = None,
    ) -> None:
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.extractor = create_feature_extractor(
            backbone, return_nodes={"layer2": "layer2", "layer3": "layer3"}
        ).to(self.device)
        self.extractor.eval()
        for parameter in self.extractor.parameters():
            parameter.requires_grad_(False)
        self.memory_bank = (
            memory_bank.detach().cpu().float() if memory_bank is not None else None
        )
        self.threshold = float(threshold)
        self._memory_device: Tensor | None = None

    @staticmethod
    def prepare_image(image: Image.Image) -> Image.Image:
        return image.convert("RGB").resize(
            DeepPatchCore.image_size, Image.Resampling.LANCZOS
        )

    @staticmethod
    def _normalize_batch(batch: Tensor) -> Tensor:
        mean = torch.tensor([0.485, 0.456, 0.406], device=batch.device)[None, :, None, None]
        std = torch.tensor([0.229, 0.224, 0.225], device=batch.device)[None, :, None, None]
        return (batch - mean) / std

    def _images_to_batch(self, images: list[Image.Image]) -> Tensor:
        arrays = []
        for image in images:
            prepared = self.prepare_image(image)
            array = np.asarray(prepared, dtype=np.float32) / 255.0
            arrays.append(torch.from_numpy(array).permute(2, 0, 1))
        batch = torch.stack(arrays).to(self.device, non_blocking=True)
        return self._normalize_batch(batch)

    @torch.inference_mode()
    def extract_embeddings(self, images: list[Image.Image]) -> Tensor:
        batch = self._images_to_batch(images)
        features = self.extractor(batch)
        layer2 = functional.avg_pool2d(features["layer2"], kernel_size=3, stride=1, padding=1)
        layer3 = functional.avg_pool2d(features["layer3"], kernel_size=3, stride=1, padding=1)
        layer3 = functional.interpolate(
            layer3, size=layer2.shape[-2:], mode="bilinear", align_corners=False
        )
        combined = torch.cat([layer2, layer3], dim=1)
        combined = functional.normalize(combined, p=2, dim=1)
        return combined.permute(0, 2, 3, 1).reshape(-1, combined.shape[1]).cpu()

    def fit(
        self,
        train_paths: list[Path],
        validation_paths: list[Path],
        batch_size: int = 16,
        candidate_size: int = 14000,
        coreset_size: int = 900,
        seed: int = 42,
    ) -> dict[str, float | int | str]:
        started = perf_counter()
        embedding_batches = []
        for start in range(0, len(train_paths), batch_size):
            images = [Image.open(path).convert("RGB") for path in train_paths[start : start + batch_size]]
            embedding_batches.append(self.extract_embeddings(images))
        embeddings = torch.cat(embedding_batches, dim=0)
        self.memory_bank = _approximate_kcenter_coreset(
            embeddings,
            candidate_size=min(candidate_size, len(embeddings)),
            coreset_size=min(coreset_size, candidate_size, len(embeddings)),
            seed=seed,
            device=self.device,
        )
        self._memory_device = None

        validation_scores = self.score_paths(validation_paths, batch_size=batch_size)
        self.threshold = float(max(validation_scores) * 1.05)
        return {
            "backbone": "resnet18_imagenet",
            "layers": "layer2,layer3",
            "device": str(self.device),
            "train_images": len(train_paths),
            "validation_images": len(validation_paths),
            "candidate_patches": min(candidate_size, len(embeddings)),
            "memory_bank_patches": len(self.memory_bank),
            "feature_dimensions": self.feature_dimensions,
            "threshold": round(self.threshold, 6),
            "training_seconds": round(perf_counter() - started, 3),
        }

    def _memory(self) -> Tensor:
        if self.memory_bank is None:
            raise RuntimeError("model has not been fitted")
        if self._memory_device is None or self._memory_device.device != self.device:
            self._memory_device = self.memory_bank.to(self.device)
        return self._memory_device

    @torch.inference_mode()
    def _score_images(self, images: list[Image.Image]) -> tuple[np.ndarray, np.ndarray]:
        embeddings = self.extract_embeddings(images).to(self.device)
        distances = torch.cdist(embeddings, self._memory())
        patch_scores = distances.min(dim=1).values
        patch_scores = patch_scores.reshape(len(images), self.grid_size, self.grid_size)
        image_scores = patch_scores.flatten(1).max(dim=1).values
        return image_scores.cpu().numpy(), patch_scores.cpu().numpy()

    def score_paths(self, paths: list[Path], batch_size: int = 12) -> list[float]:
        scores: list[float] = []
        for start in range(0, len(paths), batch_size):
            images = [Image.open(path).convert("RGB") for path in paths[start : start + batch_size]]
            batch_scores, _ = self._score_images(images)
            scores.extend(float(value) for value in batch_scores)
        return scores

    def predict(self, image: Image.Image) -> DeepPatchPrediction:
        started = perf_counter()
        prepared = self.prepare_image(image)
        scores, maps = self._score_images([prepared])
        raw_score = float(scores[0])
        coarse_map = torch.from_numpy(maps[0])[None, None]
        full_map = functional.interpolate(
            coarse_map,
            size=self.image_size,
            mode="bilinear",
            align_corners=False,
        )[0, 0].numpy()
        is_defect = raw_score > self.threshold
        overlay = render_deep_overlay(prepared, full_map, self.threshold, is_defect)
        anomaly_score = min(100.0, raw_score / max(self.threshold, 1e-8) * 50.0)
        return DeepPatchPrediction(
            is_defect=is_defect,
            anomaly_score=round(anomaly_score, 1),
            raw_score=round(raw_score, 6),
            threshold=round(self.threshold, 6),
            latency_ms=round((perf_counter() - started) * 1000, 1),
            anomaly_map=full_map,
            overlay=overlay,
            prepared_image=prepared,
        )

    def save(self, path: str | Path, metadata: dict[str, object] | None = None) -> Path:
        if self.memory_bank is None:
            raise RuntimeError("model has not been fitted")
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "memory_bank": self.memory_bank,
                "threshold": self.threshold,
                "metadata": metadata or {},
            },
            output,
        )
        return output

    @classmethod
    def load(cls, path: str | Path, device: str | None = None) -> "DeepPatchCore":
        payload = torch.load(Path(path), map_location="cpu", weights_only=True)
        return cls(
            memory_bank=payload["memory_bank"],
            threshold=float(payload["threshold"]),
            device=device,
        )


def _approximate_kcenter_coreset(
    embeddings: Tensor,
    candidate_size: int,
    coreset_size: int,
    seed: int,
    device: torch.device,
) -> Tensor:
    generator = torch.Generator().manual_seed(seed)
    candidate_indices = torch.randperm(len(embeddings), generator=generator)[:candidate_size]
    candidates = embeddings[candidate_indices].to(device)
    projection = torch.randn(
        candidates.shape[1], 64, generator=generator, dtype=candidates.dtype
    ).to(device) / np.sqrt(64)
    projected = functional.normalize(candidates @ projection, p=2, dim=1)

    selected = [0]
    min_distances = ((projected - projected[0]) ** 2).sum(dim=1)
    for _ in range(1, coreset_size):
        next_index = int(torch.argmax(min_distances).item())
        selected.append(next_index)
        distance = ((projected - projected[next_index]) ** 2).sum(dim=1)
        min_distances = torch.minimum(min_distances, distance)
    return candidates[selected].detach().cpu().float()


def render_deep_overlay(
    image: Image.Image,
    anomaly_map: np.ndarray,
    threshold: float,
    is_defect: bool,
) -> Image.Image:
    intensity = np.clip(anomaly_map / max(threshold * 1.5, 1e-8), 0, 1)
    heatmap = np.zeros((*intensity.shape, 4), dtype=np.uint8)
    heatmap[..., 0] = np.uint8(255 * intensity)
    heatmap[..., 1] = np.uint8(170 * np.power(intensity, 1.8))
    heatmap[..., 2] = np.uint8(20 * intensity)
    heatmap[..., 3] = np.uint8(210 * intensity)
    overlay = Image.alpha_composite(image.convert("RGBA"), Image.fromarray(heatmap))

    mask = anomaly_map > threshold
    if mask.any():
        ys, xs = np.where(mask)
        box = (
            max(0, int(xs.min()) - 5),
            max(0, int(ys.min()) - 5),
            min(image.width - 1, int(xs.max()) + 5),
            min(image.height - 1, int(ys.max()) + 5),
        )
        draw = ImageDraw.Draw(overlay)
        draw.rectangle(box, outline=(255, 55, 65, 255), width=4)
        label = "DEEP DEFECT" if is_defect else "DEEP CHECK"
        draw.rounded_rectangle(
            (box[0], max(0, box[1] - 28), min(image.width - 1, box[0] + 126), box[1]),
            radius=5,
            fill=(195, 28, 42, 240),
        )
        draw.text((box[0] + 7, max(2, box[1] - 22)), label, fill="white")
    return overlay.convert("RGB")


def load_mvtec_bottle(root: str | Path) -> tuple[list[Path], list[MVTecSample]]:
    bottle = Path(root)
    train_paths = sorted((bottle / "train" / "good").glob("*.png"))
    samples: list[MVTecSample] = []
    for defect_dir in sorted((bottle / "test").iterdir()):
        if not defect_dir.is_dir():
            continue
        defect_type = defect_dir.name
        for path in sorted(defect_dir.glob("*.png")):
            label = 0 if defect_type == "good" else 1
            mask_path = None
            if label:
                candidate = bottle / "ground_truth" / defect_type / f"{path.stem}_mask.png"
                mask_path = candidate if candidate.exists() else None
            samples.append(MVTecSample(path, defect_type, label, mask_path))
    return train_paths, samples


def binary_auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    truth = np.asarray(labels, dtype=np.int8).reshape(-1)
    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    order = np.argsort(-values)
    sorted_truth = truth[order]
    positives = int((truth == 1).sum())
    negatives = int((truth == 0).sum())
    if positives == 0 or negatives == 0:
        return 0.0
    tpr = np.concatenate(([0.0], np.cumsum(sorted_truth == 1) / positives, [1.0]))
    fpr = np.concatenate(([0.0], np.cumsum(sorted_truth == 0) / negatives, [1.0]))
    return float(np.trapezoid(tpr, fpr))
