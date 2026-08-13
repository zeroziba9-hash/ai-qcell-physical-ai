from __future__ import annotations

from math import cos, pi, sin
from pathlib import Path

import imageio.v2 as imageio
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "docs" / "videos" / "ai_qcell_portfolio_demo.mp4"
COVER = ROOT / "docs" / "images" / "demo_video_cover.png"
WIDTH, HEIGHT = 1280, 720
FPS = 15
SCENES = [0, 5, 12, 20, 28, 35, 42]


def font(size: int, bold: bool = False):
    candidates = [
        Path("C:/Windows/Fonts/malgunbd.ttf" if bold else "C:/Windows/Fonts/malgun.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size)
    return ImageFont.load_default()


FONTS = {
    "hero": font(58, True),
    "title": font(38, True),
    "heading": font(24, True),
    "body": font(21),
    "small": font(16),
    "metric": font(34, True),
}


def ease(value: float) -> float:
    value = min(1.0, max(0.0, value))
    return value * value * (3 - 2 * value)


def base_frame() -> Image.Image:
    image = Image.new("RGB", (WIDTH, HEIGHT), "#050b16")
    draw = ImageDraw.Draw(image)
    for y in range(HEIGHT):
        mix = y / HEIGHT
        draw.line((0, y, WIDTH, y), fill=(5, int(11 + 8 * mix), int(22 + 17 * mix)))
    for x in range(0, WIDTH, 80):
        draw.line((x, 0, x - 260, HEIGHT), fill="#0b2032", width=1)
    return image


def header(draw: ImageDraw.ImageDraw, section: str, t: float) -> None:
    draw.text((54, 32), "AI-QCELL", font=FONTS["heading"], fill="#67e8f9")
    draw.text((210, 38), section, font=FONTS["small"], fill="#94a3b8")
    draw.rounded_rectangle((54, 677, 1226, 683), radius=3, fill="#172554")
    draw.rounded_rectangle((54, 677, 54 + int(1172 * min(1, t / 42)), 683), radius=3, fill="#22d3ee")
    draw.text((1137, 647), f"{t:04.1f} / 42s", font=FONTS["small"], fill="#64748b")


def draw_text_center(draw: ImageDraw.ImageDraw, text: str, y: int, used_font, fill: str) -> None:
    box = draw.textbbox((0, 0), text, font=used_font)
    draw.text(((WIDTH - (box[2] - box[0])) / 2, y), text, font=used_font, fill=fill)


def fit_image(path: Path, box: tuple[int, int, int, int], scale: float = 1.0) -> Image.Image:
    source = Image.open(path).convert("RGB")
    box_width, box_height = box[2] - box[0], box[3] - box[1]
    ratio = max(box_width / source.width, box_height / source.height) * scale
    resized = source.resize((int(source.width * ratio), int(source.height * ratio)), Image.Resampling.LANCZOS)
    left = max(0, (resized.width - box_width) // 2)
    top = max(0, (resized.height - box_height) // 2)
    return resized.crop((left, top, left + box_width, top + box_height))


def scene_intro(local_t: float, global_t: float) -> Image.Image:
    image = base_frame()
    draw = ImageDraw.Draw(image)
    header(draw, "PHYSICAL AI PORTFOLIO", global_t)
    glow = int(95 + 60 * sin(local_t * 2.3))
    draw.ellipse((840, 95, 1230, 485), outline=(34, 211, 238, glow), width=4)
    draw.ellipse((900, 155, 1170, 425), outline="#7c3aed", width=3)
    draw_text_center(draw, "AI-QCell", 170, FONTS["hero"], "#f8fafc")
    draw_text_center(draw, "스마트팩토리 비전 검사 · 자동 선별 시스템", 252, FONTS["title"], "#67e8f9")
    draw_text_center(draw, "Deep PatchCore  ×  MVTec AD  ×  ROS2", 320, FONTS["heading"], "#c4b5fd")
    labels = [("Image AUROC", "1.000"), ("Pixel AUROC", "0.978"), ("Test images", "83")]
    for index, (label, value) in enumerate(labels):
        x = 234 + index * 280
        draw.rounded_rectangle((x, 425, x + 250, 552), radius=20, fill="#0f172a", outline="#1e3a5f", width=2)
        draw.text((x + 20, 449), label, font=FONTS["small"], fill="#94a3b8")
        draw.text((x + 20, 483), value, font=FONTS["metric"], fill="#f8fafc")
    return image


def scene_architecture(local_t: float, global_t: float) -> Image.Image:
    image = base_frame()
    draw = ImageDraw.Draw(image)
    header(draw, "01 · END-TO-END ARCHITECTURE", global_t)
    draw.text((54, 95), "산업 이미지에서 물리적 선별 동작까지", font=FONTS["title"], fill="#f8fafc")
    boxes = [
        (58, "CAMERA", "제품 이미지", "/qcell/camera/product", "#0891b2"),
        (360, "DEEP AI", "PatchCore 추론", "ResNet18 · GPU", "#7c3aed"),
        (662, "DECISION", "PASS / REJECT", "/inspection/result", "#d97706"),
        (964, "ACTUATOR", "자동 선별기", "ROS2 Action", "#e11d48"),
    ]
    visible = int(local_t / 1.1) + 1
    for index, (x, title, subtitle, detail, color) in enumerate(boxes):
        alpha = ease(local_t - index * 0.65)
        offset = int((1 - alpha) * 60)
        draw.rounded_rectangle((x, 250 + offset, x + 252, 485 + offset), radius=24, fill="#0f172a", outline=color, width=4)
        draw.ellipse((x + 24, 278 + offset, x + 70, 324 + offset), fill=color)
        draw.text((x + 87, 280 + offset), title, font=FONTS["heading"], fill="#f8fafc")
        draw.text((x + 24, 360 + offset), subtitle, font=FONTS["body"], fill="#cbd5e1")
        draw.text((x + 24, 410 + offset), detail, font=FONTS["small"], fill="#7dd3fc")
        if index < 3 and visible > index:
            draw.line((x + 255, 365, x + 294, 365), fill="#38bdf8", width=5)
            draw.polygon(((x + 294, 365), (x + 280, 356), (x + 280, 374)), fill="#38bdf8")
    draw.text((58, 550), "Topic", font=FONTS["small"], fill="#67e8f9")
    draw.text((127, 550), "실시간 데이터 전달", font=FONTS["small"], fill="#94a3b8")
    draw.text((390, 550), "Action", font=FONTS["small"], fill="#fda4af")
    draw.text((467, 550), "Goal → Feedback → Result", font=FONTS["small"], fill="#94a3b8")
    return image


def scene_visual(local_t: float, global_t: float, path: Path, title: str, caption: str, section: str) -> Image.Image:
    image = base_frame()
    draw = ImageDraw.Draw(image)
    header(draw, section, global_t)
    draw.text((54, 88), title, font=FONTS["title"], fill="#f8fafc")
    draw.text((56, 140), caption, font=FONTS["body"], fill="#94a3b8")
    zoom = 1.0 + 0.035 * ease(local_t / 8)
    preview = fit_image(path, (54, 190, 1226, 626), zoom)
    preview = ImageEnhance.Contrast(preview).enhance(1.03)
    image.paste(preview, (54, 190))
    draw.rounded_rectangle((54, 190, 1226, 626), radius=14, outline="#263f55", width=3)
    scan_x = 54 + int((1172 * ((local_t * 0.17) % 1)))
    draw.line((scan_x, 193, scan_x, 623), fill="#22d3ee", width=3)
    return image


def scene_twin(local_t: float, global_t: float) -> Image.Image:
    image = base_frame()
    draw = ImageDraw.Draw(image)
    header(draw, "05 · ROS2 DIGITAL TWIN", global_t)
    draw.text((54, 88), "AI 판정과 자동 선별 액추에이터 동기화", font=FONTS["title"], fill="#f8fafc")
    draw.rounded_rectangle((54, 178, 1226, 580), radius=22, fill="#0b1423", outline="#223650", width=3)
    draw.rounded_rectangle((80, 355, 1195, 448), radius=43, fill="#25334a", outline="#64748b", width=7)
    offset = int(local_t * 80) % 66
    for x in range(90 - offset, 1190, 66):
        draw.line((x, 363, x + 34, 440), fill="#475569", width=7)
    scan_x = 370
    draw.rounded_rectangle((332, 208, 408, 285), radius=12, fill="#334155", outline="#0ea5e9", width=4)
    draw.ellipse((351, 224, 389, 262), fill="#020617", outline="#22d3ee", width=8)
    draw.line((scan_x, 285, scan_x, 353), fill="#22d3ee", width=5)
    phase = (local_t / 7) % 1
    if phase < 0.52:
        x = 90 + (660 - 90) * ease(phase / 0.52)
        y = 345
    else:
        p = ease((phase - 0.52) / 0.48)
        x = 660 + 230 * p
        y = 345 + 165 * p
    gate_progress = min(1, max(0, (phase - 0.42) / 0.12)) * min(1, max(0, (0.92 - phase) / 0.15))
    angle = -55 * gate_progress * pi / 180
    gate_x, gate_y = 670, 338
    draw.line((gate_x, gate_y, gate_x + 112 * sin(angle), gate_y + 112 * cos(angle)), fill="#f59e0b", width=16)
    draw.line((680, 432, 905, 565), fill="#7f1d1d", width=50)
    draw.rectangle((850, 505, 1000, 580), fill="#450a0a", outline="#ef4444", width=5)
    draw.text((873, 525), "REJECT BIN", font=FONTS["small"], fill="#fca5a5")
    draw.rounded_rectangle((int(x), int(y), int(x + 58), int(y + 78)), radius=18, fill="#93c5fd", outline="#e0f2fe", width=4)
    draw.text((1005, 215), "ROS2 ACTION", font=FONTS["small"], fill="#94a3b8")
    progress = int(min(100, max(0, (phase - 0.42) / 0.5 * 100)))
    draw.text((1005, 250), f"{progress:03d}%", font=FONTS["metric"], fill="#fb7185")
    draw.rounded_rectangle((1005, 305, 1165, 323), radius=9, fill="#1e293b")
    draw.rounded_rectangle((1005, 305, 1005 + int(1.6 * progress), 323), radius=9, fill="#e11d48")
    if local_t > 5.2:
        draw.rounded_rectangle((180, 598, 1100, 654), radius=16, fill="#082f49", outline="#0e7490")
        draw_text_center(draw, "실시간 검사 · ROS2 · 디지털 트윈 · Docker · CI", 610, FONTS["heading"], "#cffafe")
    return image


def render_frame(t: float) -> Image.Image:
    if t < SCENES[1]:
        return scene_intro(t, t)
    if t < SCENES[2]:
        return scene_architecture(t - SCENES[1], t)
    if t < SCENES[3]:
        return scene_visual(
            t - SCENES[2], t, ROOT / "docs/images/qcell_vision_demo.png",
            "결함 위치를 설명하는 AI", "정상 기준 · 검사 대상 · 히트맵 · REJECT 명령", "02 · VISUAL INSPECTION",
        )
    if t < SCENES[4]:
        return scene_visual(
            t - SCENES[3], t, ROOT / "docs/images/model_evaluation.png",
            "정상 이미지만 학습하는 이상 탐지", "Patch Memory baseline · Accuracy 98.21% · Recall 100%", "03 · ANOMALY BASELINE",
        )
    if t < SCENES[5]:
        return scene_visual(
            t - SCENES[4], t, ROOT / "docs/images/deep_patchcore_bottle_evaluation.png",
            "MVTec AD 실제 산업 데이터 평가", "Image AUROC 1.000 · Pixel AUROC 0.978 · 83 test images", "04 · DEEP PATCHCORE",
        )
    return scene_twin(t - SCENES[5], t)


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(str(OUTPUT), fps=FPS, codec="libx264", quality=8, macro_block_size=1)
    try:
        for index in range(SCENES[-1] * FPS):
            frame = render_frame(index / FPS)
            writer.append_data(np.asarray(frame))
    finally:
        writer.close()
    cover = render_frame(1.8)
    overlay = Image.new("RGBA", cover.size, (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    overlay_draw.ellipse((575, 530, 705, 660), fill=(8, 47, 73, 230), outline="#67e8f9", width=4)
    overlay_draw.polygon(((625, 563), (625, 627), (677, 595)), fill="#f8fafc")
    cover = Image.alpha_composite(cover.convert("RGBA"), overlay).convert("RGB")
    cover.save(COVER, quality=94)
    print(OUTPUT)
    print(COVER)


if __name__ == "__main__":
    main()
