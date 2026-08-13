from __future__ import annotations

from io import BytesIO
import json
from pathlib import Path

import streamlit as st
from PIL import Image

from qcell.patch_memory import PatchMemoryDetector
from qcell.vision import generate_demo_pair


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "patch_memory_demo.npz"
METADATA_PATH = ROOT / "models" / "patch_memory_demo.json"
REPORT_PATH = ROOT / "docs" / "results" / "evaluation_report.json"

st.set_page_config(page_title="AI-QCell Trained Model", page_icon="🧠", layout="wide")
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


@st.cache_resource
def load_model() -> PatchMemoryDetector:
    return PatchMemoryDetector.load(MODEL_PATH)


st.title("🧠 학습형 Patch Memory 검사")
st.caption("정상 제품만 학습한 비지도 모델이 기준 이미지 없이 결함을 탐지합니다.")

if not MODEL_PATH.exists():
    st.error("학습 모델이 없습니다. `python -m scripts.train_patch_memory`를 실행하세요.")
    st.stop()

model = load_model()

settings, input_area = st.columns([0.34, 0.66], gap="large")
with settings:
    demo_type = st.selectbox(
        "검사 시나리오",
        ["scratch", "crack", "missing_part", "normal"],
        format_func=lambda value: {
            "scratch": "표면 스크래치",
            "crack": "균열",
            "missing_part": "부품 누락",
            "normal": "정상 제품",
        }[value],
    )
    st.markdown(
        """
        **학습 조건**

        - 정상 제품 이미지 40장
        - 이미지당 256개 패치
        - 패치 특징 8차원
        - 위치별 최근접 정상 메모리 비교
        """
    )
    st.info("기준 이미지 없이 학습된 정상 패턴과 비교합니다.")

with input_area:
    target_file = st.file_uploader(
        "검사 이미지 업로드 (선택)", type=["png", "jpg", "jpeg"], key="trained_target"
    )
    _, demo_target = generate_demo_pair(demo_type)
    target = Image.open(target_file).convert("RGB") if target_file else demo_target

prediction = model.predict(target)

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("AI 판정", "불량 · REJECT" if prediction.is_defect else "정상 · PASS")
c2.metric("이상 점수", f"{prediction.anomaly_score:.1f}")
c3.metric("원시 점수", f"{prediction.raw_score:.3f}")
c4.metric("결함 패치", f"{prediction.defect_ratio * 100:.2f}%")
c5.metric("추론 시간", f"{prediction.latency_ms:.1f} ms")

left, right = st.columns(2)
with left:
    st.markdown("#### 검사 대상")
    st.image(target, width="stretch")
with right:
    st.markdown("#### 학습 모델 결함 히트맵")
    st.image(prediction.overlay, width="stretch")

if prediction.is_defect:
    st.error(
        f"REJECT · 원시 점수 {prediction.raw_score:.3f}가 학습 임계값 "
        f"{prediction.threshold:.3f}을 초과했습니다."
    )
else:
    st.success(
        f"PASS_THROUGH · 원시 점수 {prediction.raw_score:.3f}가 학습 임계값 "
        f"{prediction.threshold:.3f} 이하입니다."
    )

buffer = BytesIO()
prediction.overlay.save(buffer, format="PNG")
st.download_button(
    "AI 검사 결과 다운로드",
    data=buffer.getvalue(),
    file_name="qcell_trained_model_result.png",
    mime="image/png",
)

st.markdown("---")
st.subheader("모델 평가 결과")
if REPORT_PATH.exists():
    report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    metrics = report["metrics"]
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("AUROC", f"{metrics['auroc']:.3f}")
    m2.metric("F1", f"{metrics['f1']:.3f}")
    m3.metric("Precision", f"{metrics['precision']:.3f}")
    m4.metric("Recall", f"{metrics['recall']:.3f}")
    st.image(ROOT / "docs" / "images" / "model_evaluation.png", width="stretch")

with st.expander("모델 메타데이터와 한계"):
    if METADATA_PATH.exists():
        st.json(json.loads(METADATA_PATH.read_text(encoding="utf-8")))
    st.warning(
        "현재 성능은 정렬된 합성 제품 평가 결과이며 MVTec AD 벤치마크와 직접 비교할 수 없습니다. "
        "다음 단계에서 실제 산업 데이터와 사전학습 딥러닝 특징을 적용합니다."
    )
