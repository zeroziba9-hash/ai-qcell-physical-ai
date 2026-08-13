from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from qcell.vision import generate_demo_pair, inspect_against_reference


def main() -> None:
    reference, target = generate_demo_pair("scratch")
    result = inspect_against_reference(reference, target)

    canvas = Image.new("RGB", (1680, 720), "#07111f")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=28)
    small = ImageFont.load_default(size=20)

    draw.text((55, 35), "AI-QCELL  |  PHYSICAL AI QUALITY INSPECTION", fill="#eef7ff", font=font)
    draw.text((55, 80), "Reference-to-result visual inspection pipeline", fill="#75dce5", font=small)

    panels = [
        ("NORMAL REFERENCE", reference),
        ("INSPECTION TARGET", target),
        ("AI DEFECT HEATMAP", result.overlay),
    ]
    for index, (label, image) in enumerate(panels):
        x = 55 + index * 540
        draw.rounded_rectangle((x, 135, x + 500, 520), radius=18, fill="#0e1a29", outline="#263f55", width=2)
        draw.text((x + 22, 155), label, fill="#d8e8f5", font=small)
        preview = image.resize((456, 266), Image.Resampling.LANCZOS)
        canvas.paste(preview, (x + 22, 205))

    status = "DEFECT / REJECT" if result.is_defect else "NORMAL / PASS"
    metrics = [
        f"RESULT  {status}",
        f"ANOMALY SCORE  {result.anomaly_score:.1f}/100",
        f"DEFECT AREA  {result.defect_ratio * 100:.2f}%",
        f"LATENCY  {result.latency_ms:.1f} ms",
    ]
    for index, metric in enumerate(metrics):
        x = 55 + index * 395
        draw.rounded_rectangle((x, 575, x + 355, 655), radius=15, fill="#111f31", outline="#2d4961", width=2)
        draw.text((x + 18, 603), metric, fill="#f1f7fb", font=small)

    output = Path("docs/images/qcell_vision_demo.png")
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, optimize=True)
    print(output.resolve())


if __name__ == "__main__":
    main()
