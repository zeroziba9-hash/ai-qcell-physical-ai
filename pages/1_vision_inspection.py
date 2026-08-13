from __future__ import annotations

from io import BytesIO

import streamlit as st
from PIL import Image

from qcell.vision import generate_demo_pair, inspect_against_reference


st.set_page_config(page_title="AI-QCell Vision", page_icon="🔍", layout="wide")

st.markdown(
    """
    <style>
    .block-container {padding-top: 1.4rem; padding-bottom: 3rem;}
    [data-testid="stMetric"] {
        background: linear-gradient(145deg, #111827, #0b1220);
        border: 1px solid #263244;
        padding: 16px;
        border-radius: 14px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🔍 AI 비전 불량검사")
st.caption("정상 기준 이미지와 검사 대상을 비교해 결함 위치와 제거 명령을 생성합니다.")

control_col, upload_col = st.columns([0.36, 0.64], gap="large")
with control_col:
    demo_type = st.selectbox(
        "데모 제품",
        options=["scratch", "crack", "missing_part", "normal"],
        format_func=lambda value: {
            "scratch": "표면 스크래치",
            "crack": "균열",
            "missing_part": "부품 누락",
            "normal": "정상 제품",
        }[value],
    )
    pixel_threshold = st.slider(
        "픽셀 차이 민감도",
        min_value=0.02,
        max_value=0.30,
        value=0.08,
        step=0.01,
        help="낮을수록 작은 변화도 결함 후보로 표시합니다.",
    )
    reject_ratio = st.slider(
        "불량 판정 면적 기준(%)",
        min_value=0.01,
        max_value=3.00,
        value=0.20,
        step=0.01,
    )
    st.info("현재는 기준 이미지 비교 방식의 MVP입니다. 다음 단계에서 PatchCore 모델로 교체합니다.")

with upload_col:
    reference_file = st.file_uploader(
        "정상 기준 이미지 (선택)", type=["png", "jpg", "jpeg"], key="reference"
    )
    target_file = st.file_uploader(
        "검사할 제품 이미지 (선택)", type=["png", "jpg", "jpeg"], key="target"
    )

demo_reference, demo_target = generate_demo_pair(demo_type)
reference = Image.open(reference_file).convert("RGB") if reference_file else demo_reference
target = Image.open(target_file).convert("RGB") if target_file else demo_target

result = inspect_against_reference(
    reference=reference,
    target=target,
    pixel_threshold=pixel_threshold,
    reject_ratio=reject_ratio / 100,
)

result_col1, result_col2, result_col3, result_col4 = st.columns(4)
result_col1.metric("검사 결과", "불량 · REJECT" if result.is_defect else "정상 · PASS")
result_col2.metric("이상 점수", f"{result.anomaly_score:.1f} / 100")
result_col3.metric("결함 면적", f"{result.defect_ratio * 100:.2f}%")
result_col4.metric("추론 시간", f"{result.latency_ms:.1f} ms")

image_col1, image_col2, image_col3 = st.columns(3)
with image_col1:
    st.markdown("#### 정상 기준")
    st.image(reference, width="stretch")
with image_col2:
    st.markdown("#### 검사 대상")
    st.image(target, width="stretch")
with image_col3:
    st.markdown("#### AI 결함 히트맵")
    st.image(result.overlay, width="stretch")

if result.is_defect:
    st.error("REJECT 명령 생성 · 불량품을 제거 라인으로 분류합니다.")
else:
    st.success("PASS_THROUGH 명령 생성 · 정상 제품을 다음 공정으로 이동합니다.")

overlay_buffer = BytesIO()
result.overlay.save(overlay_buffer, format="PNG")
st.download_button(
    "검사 결과 이미지 다운로드",
    data=overlay_buffer.getvalue(),
    file_name="qcell_inspection_result.png",
    mime="image/png",
)
st.markdown("---")
st.caption("포트폴리오 시각 자료: 원본 비교 · 결함 히트맵 · 정량 지표 · 액추에이터 명령")
