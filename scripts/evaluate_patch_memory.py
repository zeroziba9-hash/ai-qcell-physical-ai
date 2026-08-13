from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from qcell.patch_memory import (
    PatchMemoryDetector,
    classification_metrics,
    generate_evaluation_set,
)
from qcell.vision import generate_demo_pair


def _render_report(
    model: PatchMemoryDetector, report: dict[str, object], output: Path
) -> None:
    canvas = Image.new("RGB", (1680, 920), "#07111f")
    draw = ImageDraw.Draw(canvas)
    title_font = ImageFont.load_default(size=30)
    body_font = ImageFont.load_default(size=20)
    metric_font = ImageFont.load_default(size=24)

    draw.text((55, 32), "AI-QCELL  |  TRAINED PATCH MEMORY EVALUATION", fill="#eef7ff", font=title_font)
    draw.text((55, 78), "Normal-only unsupervised baseline on held-out synthetic samples", fill="#75dce5", font=body_font)

    metrics = report["metrics"]
    cards = [
        ("AUROC", f"{float(metrics['auroc']):.3f}"),
        ("F1 SCORE", f"{float(metrics['f1']):.3f}"),
        ("PRECISION", f"{float(metrics['precision']):.3f}"),
        ("RECALL", f"{float(metrics['recall']):.3f}"),
    ]
    for index, (label, value) in enumerate(cards):
        x = 55 + index * 395
        draw.rounded_rectangle((x, 125, x + 355, 225), radius=16, fill="#101f31", outline="#2d4961", width=2)
        draw.text((x + 20, 145), label, fill="#87a9bd", font=body_font)
        draw.text((x + 20, 178), value, fill="#f4f9fc", font=metric_font)

    examples = []
    for defect_type in ("normal", "scratch", "crack", "missing_part"):
        _, target = generate_demo_pair(defect_type)
        prediction = model.predict(target)
        examples.append((defect_type.upper(), prediction))

    for index, (label, prediction) in enumerate(examples):
        x = 55 + index * 395
        draw.rounded_rectangle((x, 270, x + 355, 600), radius=16, fill="#0e1a29", outline="#263f55", width=2)
        draw.text((x + 18, 290), label, fill="#e4f0f8", font=body_font)
        preview = prediction.overlay.resize((319, 186), Image.Resampling.LANCZOS)
        canvas.paste(preview, (x + 18, 335))
        decision = "REJECT" if prediction.is_defect else "PASS"
        color = "#ff6b78" if prediction.is_defect else "#56e0b2"
        draw.text((x + 18, 545), f"{decision}  SCORE {prediction.anomaly_score:.1f}", fill=color, font=body_font)

    confusion = metrics["confusion_matrix"]
    tn, fp = confusion[0]
    fn, tp = confusion[1]
    draw.rounded_rectangle((55, 650, 800, 855), radius=18, fill="#0e1a29", outline="#263f55", width=2)
    draw.text((80, 675), "CONFUSION MATRIX", fill="#e4f0f8", font=metric_font)
    matrix_text = [
        f"TRUE NORMAL   {tn:>3}     FALSE REJECT  {fp:>3}",
        f"MISSED DEFECT {fn:>3}     TRUE DEFECT   {tp:>3}",
        f"TOTAL SAMPLES {int(metrics['sample_count']):>3}",
    ]
    for row, text in enumerate(matrix_text):
        draw.text((80, 725 + row * 35), text, fill="#bed1de", font=body_font)

    draw.rounded_rectangle((845, 650, 1625, 855), radius=18, fill="#0e1a29", outline="#263f55", width=2)
    draw.text((870, 675), "INTERPRETATION", fill="#e4f0f8", font=metric_font)
    notes = [
        "- Trained with normal product images only",
        "- Patch-level nearest-memory anomaly scoring",
        "- Heatmap and reject decision generated together",
        "- Synthetic baseline; MVTec deep features are next",
    ]
    for row, text in enumerate(notes):
        draw.text((870, 725 + row * 32), text, fill="#bed1de", font=body_font)

    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, optimize=True)


def main() -> None:
    model = PatchMemoryDetector.load("models/patch_memory_demo.npz")
    samples = generate_evaluation_set()
    labels: list[int] = []
    scores: list[float] = []
    records: list[dict[str, object]] = []
    for defect_type, label, image in samples:
        prediction = model.predict(image)
        labels.append(label)
        scores.append(prediction.raw_score)
        records.append(
            {
                "type": defect_type,
                "label": label,
                "score": prediction.raw_score,
                "predicted_defect": prediction.is_defect,
            }
        )

    metrics = classification_metrics(labels, scores, model.image_threshold)
    per_type = {}
    for defect_type in ("normal", "scratch", "crack", "missing_part"):
        type_scores = [float(record["score"]) for record in records if record["type"] == defect_type]
        per_type[defect_type] = {
            "count": len(type_scores),
            "mean_score": round(float(np.mean(type_scores)), 4),
            "min_score": round(float(np.min(type_scores)), 4),
            "max_score": round(float(np.max(type_scores)), 4),
        }

    report: dict[str, object] = {
        "evaluation": "held_out_synthetic_products",
        "threshold": round(model.image_threshold, 6),
        "metrics": metrics,
        "per_type": per_type,
        "limitations": [
            "Synthetic aligned-product data only",
            "Not comparable to MVTec AD benchmark scores",
            "Deep pretrained features are not used yet",
        ],
    }
    report_path = Path("docs/results/evaluation_report.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    image_path = Path("docs/images/model_evaluation.png")
    _render_report(model, report, image_path)
    print(json.dumps(metrics, indent=2))
    print(f"report={report_path.resolve()}")
    print(f"visual={image_path.resolve()}")


if __name__ == "__main__":
    main()
