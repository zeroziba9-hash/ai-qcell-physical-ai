from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from qcell.deep_patchcore import DeepPatchCore, binary_auroc, load_mvtec_bottle
from qcell.patch_memory import classification_metrics


def _pixel_arrays(mask_path: Path | None, anomaly_map: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    map_small = np.asarray(
        Image.fromarray(anomaly_map.astype(np.float32)).resize((56, 56), Image.Resampling.BILINEAR),
        dtype=np.float32,
    )
    if mask_path is None:
        mask_small = np.zeros((56, 56), dtype=np.uint8)
    else:
        mask_small = np.asarray(
            Image.open(mask_path).convert("L").resize((56, 56), Image.Resampling.NEAREST),
            dtype=np.uint8,
        ) > 0
    return mask_small.reshape(-1), map_small.reshape(-1)


def _render_report(
    model: DeepPatchCore,
    samples_by_type: dict[str, object],
    report: dict[str, object],
    output: Path,
) -> None:
    canvas = Image.new("RGB", (1680, 940), "#07111f")
    draw = ImageDraw.Draw(canvas)
    title_font = ImageFont.load_default(size=30)
    body_font = ImageFont.load_default(size=20)
    metric_font = ImageFont.load_default(size=24)
    draw.text((55, 28), "AI-QCELL  |  DEEP PATCHCORE ON MVTEC AD BOTTLE", fill="#eef7ff", font=title_font)
    draw.text((55, 74), "ImageNet ResNet18 layer2/layer3 features  |  normal-only training", fill="#75dce5", font=body_font)

    metrics = report["image_metrics"]
    cards = [
        ("IMAGE AUROC", f"{float(metrics['auroc']):.3f}"),
        ("PIXEL AUROC", f"{float(report['pixel_auroc']):.3f}"),
        ("F1 SCORE", f"{float(metrics['f1']):.3f}"),
        ("RECALL", f"{float(metrics['recall']):.3f}"),
    ]
    for index, (label, value) in enumerate(cards):
        x = 55 + index * 395
        draw.rounded_rectangle((x, 120, x + 355, 220), radius=16, fill="#101f31", outline="#2d4961", width=2)
        draw.text((x + 20, 140), label, fill="#87a9bd", font=body_font)
        draw.text((x + 20, 174), value, fill="#f4f9fc", font=metric_font)

    panel_types = ["good", "broken_large", "broken_small", "contamination"]
    for index, defect_type in enumerate(panel_types):
        sample = samples_by_type[defect_type]
        prediction = model.predict(Image.open(sample.path).convert("RGB"))
        x = 55 + index * 395
        draw.rounded_rectangle((x, 265, x + 355, 610), radius=16, fill="#0e1a29", outline="#263f55", width=2)
        draw.text((x + 18, 285), defect_type.upper(), fill="#e4f0f8", font=body_font)
        preview = prediction.overlay.resize((319, 260), Image.Resampling.LANCZOS)
        canvas.paste(preview, (x + 18, 325))
        decision = "REJECT" if prediction.is_defect else "PASS"
        color = "#ff6b78" if prediction.is_defect else "#56e0b2"
        draw.text((x + 18, 588), f"{decision}  SCORE {prediction.raw_score:.3f}", fill=color, font=body_font)

    confusion = metrics["confusion_matrix"]
    tn, fp = confusion[0]
    fn, tp = confusion[1]
    draw.rounded_rectangle((55, 655, 800, 875), radius=18, fill="#0e1a29", outline="#263f55", width=2)
    draw.text((80, 680), "IMAGE-LEVEL CONFUSION MATRIX", fill="#e4f0f8", font=metric_font)
    lines = [
        f"TRUE NORMAL   {tn:>3}     FALSE REJECT  {fp:>3}",
        f"MISSED DEFECT {fn:>3}     TRUE DEFECT   {tp:>3}",
        f"TOTAL TEST IMAGES {int(metrics['sample_count']):>3}",
        f"THRESHOLD {float(report['threshold']):.4f}",
    ]
    for row, text in enumerate(lines):
        draw.text((80, 730 + row * 34), text, fill="#bed1de", font=body_font)

    draw.rounded_rectangle((845, 655, 1625, 875), radius=18, fill="#0e1a29", outline="#263f55", width=2)
    draw.text((870, 680), "PORTFOLIO EVIDENCE", fill="#e4f0f8", font=metric_font)
    notes = [
        "- 177 normal images build the memory bank",
        "- 32 held-out normals calibrate threshold",
        "- 83 real MVTec test images evaluated",
        "- Heatmap localizes defects without defect training",
        "- RTX GPU inference with pretrained CNN features",
    ]
    for row, text in enumerate(notes):
        draw.text((870, 730 + row * 30), text, fill="#bed1de", font=body_font)
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, optimize=True)


def main() -> None:
    model = DeepPatchCore.load("models/deep_patchcore_bottle.pt")
    _, samples = load_mvtec_bottle("data/mvtec-ad/bottle")
    labels: list[int] = []
    scores: list[float] = []
    pixel_labels = []
    pixel_scores = []
    per_type_scores: dict[str, list[float]] = {}
    samples_by_type = {}

    for sample in samples:
        prediction = model.predict(Image.open(sample.path).convert("RGB"))
        labels.append(sample.label)
        scores.append(prediction.raw_score)
        per_type_scores.setdefault(sample.defect_type, []).append(prediction.raw_score)
        samples_by_type.setdefault(sample.defect_type, sample)
        mask_values, map_values = _pixel_arrays(sample.mask_path, prediction.anomaly_map)
        pixel_labels.append(mask_values)
        pixel_scores.append(map_values)

    image_metrics = classification_metrics(labels, scores, model.threshold)
    pixel_auroc = binary_auroc(np.concatenate(pixel_labels), np.concatenate(pixel_scores))
    per_type = {
        defect_type: {
            "count": len(values),
            "mean_score": round(float(np.mean(values)), 6),
            "min_score": round(float(np.min(values)), 6),
            "max_score": round(float(np.max(values)), 6),
        }
        for defect_type, values in per_type_scores.items()
    }
    report: dict[str, object] = {
        "dataset": "MVTec AD",
        "category": "bottle",
        "method": "Deep PatchCore-style ResNet18 layer2/layer3 memory bank",
        "threshold": round(model.threshold, 6),
        "image_metrics": image_metrics,
        "pixel_auroc": round(pixel_auroc, 4),
        "per_type": per_type,
        "license": "CC BY-NC-SA 4.0",
    }
    report_path = Path("docs/results/deep_patchcore_bottle_report.json")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    visual_path = Path("docs/images/deep_patchcore_bottle_evaluation.png")
    _render_report(model, samples_by_type, report, visual_path)
    print(json.dumps(report, indent=2))
    print(f"report={report_path.resolve()}")
    print(f"visual={visual_path.resolve()}")


if __name__ == "__main__":
    main()
