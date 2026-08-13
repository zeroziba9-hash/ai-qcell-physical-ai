from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageOps


@dataclass(frozen=True)
class VisionInspectionResult:
    is_defect: bool
    anomaly_score: float
    defect_ratio: float
    latency_ms: float
    heatmap: Image.Image
    overlay: Image.Image


def _fit_rgb(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    return ImageOps.fit(image.convert("RGB"), size, method=Image.Resampling.LANCZOS)


def inspect_against_reference(
    reference: Image.Image,
    target: Image.Image,
    pixel_threshold: float = 0.08,
    reject_ratio: float = 0.002,
) -> VisionInspectionResult:
    """Compare a product image with a normal reference and render a defect map."""
    if not 0 < pixel_threshold < 1:
        raise ValueError("pixel_threshold must be between 0 and 1")
    if not 0 <= reject_ratio <= 1:
        raise ValueError("reject_ratio must be between 0 and 1")

    started = perf_counter()
    size = reference.size
    reference_rgb = _fit_rgb(reference, size)
    target_rgb = _fit_rgb(target, size)

    reference_array = np.asarray(reference_rgb, dtype=np.float32) / 255.0
    target_array = np.asarray(target_rgb, dtype=np.float32) / 255.0
    difference = np.mean(np.abs(target_array - reference_array), axis=2)

    difference_image = Image.fromarray(np.uint8(np.clip(difference, 0, 1) * 255))
    smoothed = np.asarray(
        difference_image.filter(ImageFilter.GaussianBlur(radius=1.4)), dtype=np.float32
    ) / 255.0
    defect_mask = smoothed >= pixel_threshold
    defect_ratio = float(defect_mask.mean())
    percentile_score = float(np.percentile(smoothed, 99.5))
    anomaly_score = min(100.0, percentile_score / max(pixel_threshold, 1e-6) * 55.0)
    is_defect = defect_ratio >= reject_ratio

    intensity = np.clip(
        (smoothed - pixel_threshold * 0.45) / max(1 - pixel_threshold, 1e-6), 0, 1
    )
    heatmap_array = np.zeros((*intensity.shape, 4), dtype=np.uint8)
    heatmap_array[..., 0] = np.uint8(255 * intensity)
    heatmap_array[..., 1] = np.uint8(190 * np.power(intensity, 1.5))
    heatmap_array[..., 2] = np.uint8(35 * intensity)
    heatmap_array[..., 3] = np.uint8(220 * intensity)
    heatmap = Image.fromarray(heatmap_array)

    overlay = Image.alpha_composite(target_rgb.convert("RGBA"), heatmap)
    if defect_mask.any():
        ys, xs = np.where(defect_mask)
        padding = 8
        box = (
            max(0, int(xs.min()) - padding),
            max(0, int(ys.min()) - padding),
            min(size[0] - 1, int(xs.max()) + padding),
            min(size[1] - 1, int(ys.max()) + padding),
        )
        draw = ImageDraw.Draw(overlay)
        draw.rectangle(box, outline=(255, 60, 60, 255), width=4)
        label = "DEFECT" if is_defect else "CHECK"
        draw.rounded_rectangle(
            (box[0], max(0, box[1] - 30), box[0] + 92, box[1]),
            radius=5,
            fill=(205, 32, 45, 240),
        )
        draw.text((box[0] + 10, max(2, box[1] - 24)), label, fill="white")

    latency_ms = (perf_counter() - started) * 1000
    return VisionInspectionResult(
        is_defect=is_defect,
        anomaly_score=round(anomaly_score, 1),
        defect_ratio=defect_ratio,
        latency_ms=round(latency_ms, 1),
        heatmap=heatmap,
        overlay=overlay.convert("RGB"),
    )


def generate_demo_pair(defect_type: str = "scratch") -> tuple[Image.Image, Image.Image]:
    """Create a clean reference and a synthetic inspection sample for the dashboard."""
    if defect_type not in {"normal", "scratch", "crack", "missing_part"}:
        raise ValueError(f"unsupported defect type: {defect_type}")

    width, height = 720, 420
    base = Image.new("RGB", (width, height), "#07111f")
    draw = ImageDraw.Draw(base)
    for y in range(height):
        shade = int(12 + 16 * y / height)
        draw.line((0, y, width, y), fill=(5, shade, shade + 15))

    draw.rounded_rectangle((95, 65, 625, 355), radius=42, fill="#506174", outline="#9fb5ca", width=4)
    draw.rounded_rectangle((135, 100, 585, 320), radius=24, fill="#1e3945", outline="#58d1d9", width=3)
    draw.rectangle((170, 138, 550, 282), fill="#243f4c", outline="#8aa9b5", width=2)
    draw.line((190, 210, 530, 210), fill="#3e5d68", width=3)

    centers = [(230, 175), (360, 175), (490, 175), (230, 248), (360, 248), (490, 248)]
    for index, (x, y) in enumerate(centers):
        color = "#43d3c6" if index % 2 == 0 else "#8b9eff"
        draw.ellipse((x - 24, y - 24, x + 24, y + 24), fill=color, outline="#dce7ef", width=3)
        draw.ellipse((x - 8, y - 8, x + 8, y + 8), fill="#10222c")

    reference = base.copy()
    target = base.copy()
    defect_draw = ImageDraw.Draw(target)
    if defect_type == "scratch":
        defect_draw.line((278, 125, 445, 294), fill="#f4f7fa", width=8)
        defect_draw.line((286, 125, 453, 294), fill="#8c2938", width=3)
    elif defect_type == "crack":
        points = [(396, 112), (382, 154), (409, 185), (387, 223), (417, 266), (401, 310)]
        defect_draw.line(points, fill="#03070b", width=11, joint="curve")
        defect_draw.line(points, fill="#ff445d", width=3, joint="curve")
    elif defect_type == "missing_part":
        x, y = centers[-1]
        defect_draw.ellipse((x - 27, y - 27, x + 27, y + 27), fill="#243f4c", outline="#243f4c")

    return reference, target
