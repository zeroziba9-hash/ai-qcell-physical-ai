from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageOps

from .vision import generate_demo_pair


@dataclass(frozen=True)
class PatchPrediction:
    is_defect: bool
    anomaly_score: float
    raw_score: float
    threshold: float
    defect_ratio: float
    latency_ms: float
    heatmap: Image.Image
    overlay: Image.Image


@dataclass
class PatchMemoryDetector:
    memory: np.ndarray
    center: np.ndarray
    scale: np.ndarray
    image_size: tuple[int, int]
    grid_size: int
    patch_threshold: float
    image_threshold: float

    @classmethod
    def fit(
        cls,
        images: list[Image.Image],
        image_size: tuple[int, int] = (256, 256),
        grid_size: int = 16,
    ) -> "PatchMemoryDetector":
        if len(images) < 3:
            raise ValueError("at least three normal images are required")

        raw_features = np.stack(
            [_extract_patch_features(image, image_size, grid_size) for image in images]
        )
        center = raw_features.mean(axis=(0, 1))
        scale = raw_features.std(axis=(0, 1)) + 1e-6
        memory = (raw_features - center) / scale

        leave_one_out_maps = []
        leave_one_out_scores = []
        for index in range(len(memory)):
            candidates = np.delete(memory, index, axis=0)
            distances = np.linalg.norm(candidates - memory[index][None, :, :], axis=2)
            patch_scores = distances.min(axis=0)
            leave_one_out_maps.append(patch_scores)
            leave_one_out_scores.append(float(np.percentile(patch_scores, 99.5)))

        all_patch_scores = np.concatenate(leave_one_out_maps)
        patch_threshold = float(np.quantile(all_patch_scores, 0.997) * 1.15 + 1e-6)
        image_threshold = float(
            max(np.quantile(leave_one_out_scores, 0.99) * 1.20, patch_threshold)
        )
        return cls(
            memory=memory.astype(np.float32),
            center=center.astype(np.float32),
            scale=scale.astype(np.float32),
            image_size=image_size,
            grid_size=grid_size,
            patch_threshold=patch_threshold,
            image_threshold=image_threshold,
        )

    def predict(self, image: Image.Image) -> PatchPrediction:
        started = perf_counter()
        target = ImageOps.fit(
            image.convert("RGB"), self.image_size, method=Image.Resampling.LANCZOS
        )
        raw_features = _extract_patch_features(target, self.image_size, self.grid_size)
        normalized = (raw_features - self.center) / self.scale
        distances = np.linalg.norm(self.memory - normalized[None, :, :], axis=2)
        patch_scores = distances.min(axis=0)
        raw_score = float(np.percentile(patch_scores, 99.5))
        is_defect = raw_score > self.image_threshold
        defect_ratio = float((patch_scores > self.patch_threshold).mean())

        score_grid = patch_scores.reshape(self.grid_size, self.grid_size)
        normalized_grid = np.clip(score_grid / max(self.image_threshold * 1.8, 1e-6), 0, 1)
        score_image = Image.fromarray(np.uint8(normalized_grid * 255)).resize(
            self.image_size, Image.Resampling.BICUBIC
        ).filter(ImageFilter.GaussianBlur(radius=4))
        intensity = np.asarray(score_image, dtype=np.float32) / 255.0

        heatmap_array = np.zeros((*intensity.shape, 4), dtype=np.uint8)
        heatmap_array[..., 0] = np.uint8(255 * intensity)
        heatmap_array[..., 1] = np.uint8(175 * np.power(intensity, 1.7))
        heatmap_array[..., 2] = np.uint8(25 * intensity)
        heatmap_array[..., 3] = np.uint8(225 * intensity)
        heatmap = Image.fromarray(heatmap_array)
        overlay = Image.alpha_composite(target.convert("RGBA"), heatmap)

        mask_grid = score_grid > self.patch_threshold
        if mask_grid.any():
            ys, xs = np.where(mask_grid)
            patch_width = self.image_size[0] / self.grid_size
            patch_height = self.image_size[1] / self.grid_size
            box = (
                max(0, int(xs.min() * patch_width) - 4),
                max(0, int(ys.min() * patch_height) - 4),
                min(self.image_size[0] - 1, int((xs.max() + 1) * patch_width) + 4),
                min(self.image_size[1] - 1, int((ys.max() + 1) * patch_height) + 4),
            )
            draw = ImageDraw.Draw(overlay)
            draw.rectangle(box, outline=(255, 55, 65, 255), width=4)
            label = "AI DEFECT" if is_defect else "AI CHECK"
            label_width = 108
            draw.rounded_rectangle(
                (box[0], max(0, box[1] - 28), box[0] + label_width, box[1]),
                radius=5,
                fill=(205, 32, 45, 240),
            )
            draw.text((box[0] + 8, max(2, box[1] - 22)), label, fill="white")

        anomaly_score = min(100.0, raw_score / max(self.image_threshold, 1e-6) * 55.0)
        return PatchPrediction(
            is_defect=is_defect,
            anomaly_score=round(anomaly_score, 1),
            raw_score=round(raw_score, 4),
            threshold=round(self.image_threshold, 4),
            defect_ratio=defect_ratio,
            latency_ms=round((perf_counter() - started) * 1000, 1),
            heatmap=heatmap,
            overlay=overlay.convert("RGB"),
        )

    def save(self, path: str | Path) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output,
            memory=self.memory,
            center=self.center,
            scale=self.scale,
            image_size=np.asarray(self.image_size),
            grid_size=np.asarray(self.grid_size),
            patch_threshold=np.asarray(self.patch_threshold),
            image_threshold=np.asarray(self.image_threshold),
        )
        return output

    @classmethod
    def load(cls, path: str | Path) -> "PatchMemoryDetector":
        with np.load(Path(path)) as data:
            return cls(
                memory=data["memory"],
                center=data["center"],
                scale=data["scale"],
                image_size=tuple(int(value) for value in data["image_size"]),
                grid_size=int(data["grid_size"]),
                patch_threshold=float(data["patch_threshold"]),
                image_threshold=float(data["image_threshold"]),
            )


def _extract_patch_features(
    image: Image.Image, image_size: tuple[int, int], grid_size: int
) -> np.ndarray:
    fitted = ImageOps.fit(
        image.convert("RGB"), image_size, method=Image.Resampling.LANCZOS
    )
    array = np.asarray(fitted, dtype=np.float32) / 255.0
    height, width, _ = array.shape
    if height % grid_size or width % grid_size:
        raise ValueError("image dimensions must be divisible by grid_size")

    gray = array.mean(axis=2)
    gradient_x = np.diff(gray, axis=1, prepend=gray[:, :1])
    gradient_y = np.diff(gray, axis=0, prepend=gray[:1, :])
    gradient = np.sqrt(gradient_x**2 + gradient_y**2)

    patch_height = height // grid_size
    patch_width = width // grid_size
    rgb_patches = array.reshape(
        grid_size, patch_height, grid_size, patch_width, 3
    ).transpose(0, 2, 1, 3, 4)
    gradient_patches = gradient.reshape(
        grid_size, patch_height, grid_size, patch_width
    ).transpose(0, 2, 1, 3)

    means = rgb_patches.mean(axis=(2, 3))
    standard_deviations = rgb_patches.std(axis=(2, 3))
    gradient_mean = gradient_patches.mean(axis=(2, 3))[..., None]
    gradient_std = gradient_patches.std(axis=(2, 3))[..., None]
    features = np.concatenate(
        [means, standard_deviations, gradient_mean, gradient_std], axis=2
    )
    return features.reshape(grid_size * grid_size, -1).astype(np.float32)


def augment_product(image: Image.Image, rng: np.random.Generator) -> Image.Image:
    output = image.convert("RGB")
    output = ImageEnhance.Brightness(output).enhance(float(rng.uniform(0.94, 1.06)))
    output = ImageEnhance.Contrast(output).enhance(float(rng.uniform(0.96, 1.04)))
    array = np.asarray(output, dtype=np.float32)
    noise = rng.normal(0, 1.5, size=array.shape)
    return Image.fromarray(np.uint8(np.clip(array + noise, 0, 255)))


def generate_normal_training_set(count: int = 40, seed: int = 42) -> list[Image.Image]:
    reference, _ = generate_demo_pair("normal")
    rng = np.random.default_rng(seed)
    return [augment_product(reference, rng) for _ in range(count)]


def generate_evaluation_set(
    normal_count: int = 20,
    defect_count_per_type: int = 12,
    seed: int = 2026,
) -> list[tuple[str, int, Image.Image]]:
    rng = np.random.default_rng(seed)
    samples: list[tuple[str, int, Image.Image]] = []
    normal, _ = generate_demo_pair("normal")
    for _ in range(normal_count):
        samples.append(("normal", 0, augment_product(normal, rng)))
    for defect_type in ("scratch", "crack", "missing_part"):
        _, defect = generate_demo_pair(defect_type)
        for _ in range(defect_count_per_type):
            samples.append((defect_type, 1, augment_product(defect, rng)))
    return samples


def classification_metrics(
    labels: list[int], scores: list[float], threshold: float
) -> dict[str, float | int | list[list[int]]]:
    truth = np.asarray(labels, dtype=np.int32)
    score_array = np.asarray(scores, dtype=np.float64)
    predictions = (score_array > threshold).astype(np.int32)
    tp = int(((truth == 1) & (predictions == 1)).sum())
    tn = int(((truth == 0) & (predictions == 0)).sum())
    fp = int(((truth == 0) & (predictions == 1)).sum())
    fn = int(((truth == 1) & (predictions == 0)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    accuracy = (tp + tn) / len(truth) if len(truth) else 0.0

    order = np.argsort(-score_array)
    sorted_truth = truth[order]
    positives = max(int((truth == 1).sum()), 1)
    negatives = max(int((truth == 0).sum()), 1)
    tpr = np.concatenate(([0.0], np.cumsum(sorted_truth == 1) / positives, [1.0]))
    fpr = np.concatenate(([0.0], np.cumsum(sorted_truth == 0) / negatives, [1.0]))
    auroc = float(np.trapezoid(tpr, fpr))
    return {
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "auroc": round(auroc, 4),
        "confusion_matrix": [[tn, fp], [fn, tp]],
        "sample_count": int(len(truth)),
    }
