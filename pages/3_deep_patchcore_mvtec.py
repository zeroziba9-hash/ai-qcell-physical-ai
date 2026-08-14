from __future__ import annotations

from io import BytesIO
import json
from pathlib import Path

import streamlit as st

from qcell.ui import inject_global_css, page_header, workflow_strip
from PIL import Image
import torch

from qcell.deep_patchcore import DeepPatchCore, load_mvtec_bottle
from qcell.model_registry import ModelRegistry


ROOT = Path(__file__).resolve().parents[1]
FALLBACK_MODEL_PATH = ROOT / "models" / "deep_patchcore_bottle.pt"
REGISTRY_ROOT = ROOT / "artifacts" / "active_learning" / "model_registry"
MODEL_PATH, MODEL_VERSION = ModelRegistry(REGISTRY_ROOT).resolve_model_path(FALLBACK_MODEL_PATH)
DATASET_PATH = ROOT / "data" / "mvtec-ad" / "bottle"
REPORT_PATH = ROOT / "docs" / "results" / "deep_patchcore_bottle_report.json"
VISUAL_PATH = ROOT / "docs" / "images" / "deep_patchcore_bottle_evaluation.png"

st.set_page_config(page_title="AI-QCell Deep PatchCore", page_icon="🧬", layout="wide")
inject_global_css()
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
def load_deep_model(path: str, modified_ns: int) -> DeepPatchCore:
    del modified_ns
    return DeepPatchCore.load(path)


page_header(
    "PRODUCTION VISION · MVTec AD",
    "Deep PatchCore 산업 결함검사",
    "ImageNet ResNet18 중간 특징과 정상 패치 메모리 뱅크로 실제 산업 결함을 탐지합니다.",
    status="PRODUCTION MODEL",
)
workflow_strip(["Feature extract", "Memory bank", "Nearest patch", "Anomaly heatmap"])

if not MODEL_PATH.exists():
    st.error("Deep PatchCore 모델이 없습니다. `python -m scripts.train_deep_patchcore`를 실행하세요.")
    st.stop()

model = load_deep_model(str(MODEL_PATH), MODEL_PATH.stat().st_mtime_ns)
_, all_samples = load_mvtec_bottle(DATASET_PATH) if DATASET_PATH.exists() else ([], [])
samples_by_type = {}
for sample in all_samples:
    samples_by_type.setdefault(sample.defect_type, []).append(sample)

controls, uploader = st.columns([0.36, 0.64], gap="large")
with controls:
    available_types = list(samples_by_type) or ["good"]
    defect_type = st.selectbox(
        "MVTec 검사 유형",
        available_types,
        format_func=lambda value: {
            "good": "정상 병",
            "broken_large": "대형 파손",
            "broken_small": "소형 파손",
            "contamination": "오염",
        }.get(value, value),
    )
    type_samples = samples_by_type.get(defect_type, [])
    sample_index = st.slider(
        "샘플 번호",
        min_value=0,
        max_value=max(0, len(type_samples) - 1),
        value=0,
        disabled=not type_samples,
    )
    st.markdown(
        """
        **모델 구성**

        - ImageNet 사전학습 ResNet18
        - `layer2` + `layer3` 특징
        - 정상 이미지 177장
        - k-center 메모리 패치 900개
        - 결함 이미지 학습 없음
        """
    )

with uploader:
    uploaded = st.file_uploader(
        "사용자 병 이미지 업로드 (선택)", type=["png", "jpg", "jpeg"], key="deep_target"
    )
    if uploaded:
        target = Image.open(uploaded).convert("RGB")
        source_label = "사용자 업로드"
    elif type_samples:
        chosen = type_samples[sample_index]
        target = Image.open(chosen.path).convert("RGB")
        source_label = f"MVTec AD / bottle / {chosen.defect_type} / {chosen.path.name}"
    else:
        st.warning("MVTec 데이터가 없습니다. 이미지를 업로드하거나 다운로드 스크립트를 실행하세요.")
        st.stop()

prediction = model.predict(target)
gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Deep AI 판정", "불량 · REJECT" if prediction.is_defect else "정상 · PASS")
c2.metric("정규화 점수", f"{prediction.anomaly_score:.1f}")
c3.metric("원시 거리", f"{prediction.raw_score:.4f}")
c4.metric("학습 임계값", f"{prediction.threshold:.4f}")
c5.metric("GPU 추론", f"{prediction.latency_ms:.1f} ms")

left, right = st.columns(2)
with left:
    st.markdown("#### 실제 검사 이미지")
    st.image(prediction.prepared_image, width="stretch")
    st.caption(source_label)
with right:
    st.markdown("#### Deep PatchCore 결함 히트맵")
    st.image(prediction.overlay, width="stretch")
    st.caption(f"실행 장치: {gpu_name}")

if prediction.is_defect:
    st.error(
        f"REJECT · 최근접 정상 패치 거리 {prediction.raw_score:.4f}가 "
        f"임계값 {prediction.threshold:.4f}을 초과했습니다."
    )
else:
    st.success(
        f"PASS_THROUGH · 최근접 정상 패치 거리 {prediction.raw_score:.4f}가 "
        f"임계값 {prediction.threshold:.4f} 이하입니다."
    )

buffer = BytesIO()
prediction.overlay.save(buffer, format="PNG")
st.download_button(
    "Deep PatchCore 결과 다운로드",
    data=buffer.getvalue(),
    file_name="qcell_deep_patchcore_result.png",
    mime="image/png",
)

st.markdown("---")
st.subheader("MVTec bottle 정량 평가")
if REPORT_PATH.exists():
    report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    metrics = report["image_metrics"]
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Image AUROC", f"{metrics['auroc']:.3f}")
    m2.metric("Pixel AUROC", f"{report['pixel_auroc']:.3f}")
    m3.metric("F1", f"{metrics['f1']:.3f}")
    m4.metric("Precision", f"{metrics['precision']:.3f}")
    m5.metric("Recall", f"{metrics['recall']:.3f}")
    st.image(VISUAL_PATH, width="stretch")

st.info(
    "MVTec AD bottle 83개 테스트 이미지 결과입니다. 데이터는 CC BY-NC-SA 4.0이며 "
    "연구·교육용 포트폴리오 범위로 사용합니다."
)
st.markdown(
    "[MVTec AD 데이터셋](https://www.mvtec.com/research-teaching/datasets/mvtec) · "
    "[Anomalib PatchCore 공식 문서](https://anomalib.readthedocs.io/en/latest/markdown/guides/reference/models/image/patchcore.html)"
)
