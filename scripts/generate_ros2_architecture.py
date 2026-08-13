from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "docs" / "images" / "ros2_pipeline_architecture.png"


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        Path("C:/Windows/Fonts/malgunbd.ttf" if bold else "C:/Windows/Fonts/malgun.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size=size)
    return ImageFont.load_default()


def arrow(draw: ImageDraw.ImageDraw, start: tuple[int, int], end: tuple[int, int]) -> None:
    draw.line((start, end), fill="#38bdf8", width=6)
    draw.polygon(
        [(end[0], end[1]), (end[0] - 18, end[1] - 11), (end[0] - 18, end[1] + 11)],
        fill="#38bdf8",
    )


def main() -> None:
    image = Image.new("RGB", (1600, 620), "#050b16")
    draw = ImageDraw.Draw(image)
    draw.text((70, 48), "AI-QCell · ROS2 자동 선별 파이프라인", font=font(38, True), fill="#f8fafc")
    draw.text(
        (72, 105),
        "실제 산업 이미지 입력 → Deep PatchCore 이상 탐지 → 판정 → 리젝트 액추에이터",
        font=font(21),
        fill="#94a3b8",
    )

    boxes = [
        (65, "01", "camera_node", "MVTec / 카메라 입력", "/qcell/camera/product", "#0e7490"),
        (450, "02", "inspection_node", "ResNet18 PatchCore", "/qcell/inspection/result", "#6d28d9"),
        (835, "03", "decision_node", "PASS / REJECT", "Topic + Action Client", "#b45309"),
        (1220, "04", "reject_action_server", "자동 선별 액추에이터", "Goal → Feedback → Result", "#be123c"),
    ]
    for x, index, node, subtitle, channel, accent in boxes:
        draw.rounded_rectangle((x, 185, x + 315, 425), radius=24, fill="#0f172a", outline=accent, width=4)
        draw.rounded_rectangle((x + 22, 207, x + 78, 254), radius=12, fill=accent)
        draw.text((x + 34, 217), index, font=font(18, True), fill="white")
        draw.text((x + 22, 282), node, font=font(23, True), fill="#f8fafc")
        draw.text((x + 22, 328), subtitle, font=font(19), fill="#cbd5e1")
        draw.rounded_rectangle((x + 20, 365, x + 295, 405), radius=10, fill="#111d33")
        draw.text((x + 32, 374), channel, font=font(15), fill="#7dd3fc")

    for x in (380, 765, 1150):
        arrow(draw, (x + 8, 305), (x + 58, 305))

    draw.rounded_rectangle((65, 480, 1535, 560), radius=18, fill="#081a25", outline="#155e75", width=2)
    draw.text((90, 500), "관찰 가능 데이터", font=font(19, True), fill="#67e8f9")
    draw.text(
        (280, 500),
        "제품 ID  ·  이상 점수/임계값  ·  GPU 추론시간  ·  Action 진행률 25/50/75/100%  ·  누적 PASS/REJECT KPI",
        font=font(18),
        fill="#dbeafe",
    )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    image.save(OUTPUT, optimize=True)
    print(OUTPUT)


if __name__ == "__main__":
    main()
