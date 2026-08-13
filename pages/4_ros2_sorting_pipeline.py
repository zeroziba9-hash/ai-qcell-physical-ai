from __future__ import annotations

from pathlib import Path

import pandas as pd
from PIL import Image
import streamlit as st
import torch

from qcell.deep_patchcore import DeepPatchCore, load_mvtec_bottle
from qcell.model_registry import ModelRegistry
from qcell.ros2_pipeline import PipelineInspection, PipelineRun, simulate_sort_pipeline


ROOT = Path(__file__).resolve().parents[1]
FALLBACK_MODEL_PATH = ROOT / "models" / "deep_patchcore_bottle.pt"
REGISTRY_ROOT = ROOT / "artifacts" / "active_learning" / "model_registry"
MODEL_PATH, MODEL_VERSION = ModelRegistry(REGISTRY_ROOT).resolve_model_path(FALLBACK_MODEL_PATH)
DATASET_PATH = ROOT / "data" / "mvtec-ad" / "bottle"

st.set_page_config(page_title="AI-QCell ROS2 Sorting", page_icon="🤖", layout="wide")
st.markdown(
    """
    <style>
    .block-container {padding-top:1.35rem; padding-bottom:3rem;}
    [data-testid="stMetric"] {background:#0f172a;border:1px solid #263244;padding:14px;border-radius:14px;}
    .pipeline {display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:1rem 0 1.4rem;}
    .stage {background:#0b1220;border:1px solid #334155;border-radius:16px;padding:18px;min-height:130px;}
    .stage.done {border-color:#10b981;background:linear-gradient(145deg,#0b1220,#06251e);}
    .stage.active {border-color:#22d3ee;box-shadow:0 0 20px #22d3ee35;}
    .stage.waiting {opacity:.45;}
    .stage .index {font-size:.76rem;color:#94a3b8;letter-spacing:.08em;}
    .stage .title {font-size:1.05rem;font-weight:700;margin:.35rem 0;}
    .stage .state {color:#67e8f9;font-size:.88rem;}
    .topic {font-family:monospace;color:#a5b4fc;font-size:.78rem;margin-top:.55rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_model(path: str, modified_ns: int) -> DeepPatchCore:
    del modified_ns
    return DeepPatchCore.load(path)


def stage_html(index: int, title: str, state: str, topic: str, css_class: str) -> str:
    return (
        f'<div class="stage {css_class}"><div class="index">STAGE {index:02d}</div>'
        f'<div class="title">{title}</div><div class="state">{state}</div>'
        f'<div class="topic">{topic}</div></div>'
    )


def render_pipeline(run: PipelineRun, current_step: int) -> None:
    inspection_done = current_step >= 3
    decision_done = current_step >= 4
    sorting_done = current_step >= len(run.events) - 1
    active_stage = 1 if current_step == 1 else 2 if not inspection_done else 3 if not decision_done else 4
    stages = [
        ("산업용 카메라", "제품 이미지 발행", "/qcell/camera/product"),
        ("Deep PatchCore", "검사 완료" if inspection_done else "GPU 추론 중", "/qcell/inspection/result"),
        ("판정 노드", run.decision if decision_done else "판정 대기", "PASS / REJECT"),
        ("자동 선별기", run.final_state if sorting_done else "액추에이터 대기", "/qcell/reject_product"),
    ]
    cards = []
    for index, (title, state, topic) in enumerate(stages, start=1):
        is_done = index < active_stage or (index == 4 and sorting_done)
        css_class = "done" if is_done else "active" if index == active_stage else "waiting"
        cards.append(stage_html(index, title, state, topic, css_class))
    st.markdown(f'<div class="pipeline">{"".join(cards)}</div>', unsafe_allow_html=True)


st.title("🤖 ROS2 자동 선별 디지털 트윈")
st.caption("MVTec 제품 투입부터 Deep PatchCore 검사, PASS/REJECT 결정, 리젝트 액추에이터까지 재현합니다.")

if not MODEL_PATH.exists() or not DATASET_PATH.exists():
    st.error("모델 또는 MVTec 데이터가 없습니다. 먼저 Deep PatchCore 학습 단계를 실행하세요.")
    st.stop()

_, samples = load_mvtec_bottle(DATASET_PATH)
samples_by_type: dict[str, list] = {}
for sample in samples:
    samples_by_type.setdefault(sample.defect_type, []).append(sample)

with st.sidebar:
    st.header("제품 투입")
    defect_type = st.selectbox(
        "샘플 유형",
        list(samples_by_type),
        format_func=lambda value: {
            "good": "정상 병",
            "broken_large": "대형 파손",
            "broken_small": "소형 파손",
            "contamination": "오염",
        }.get(value, value),
    )
    selected_samples = samples_by_type[defect_type]
    sample_index = st.slider("샘플 번호", 0, len(selected_samples) - 1, 0)
    selected = selected_samples[sample_index]
    next_number = len(st.session_state.get("ros2_history", [])) + 1
    product_id = f"QCELL-{next_number:04d}"
    run_clicked = st.button("제품 1개 투입", type="primary", width="stretch")
    if st.button("운영 기록 초기화", width="stretch"):
        st.session_state.ros2_history = []
        st.session_state.pop("ros2_latest", None)
        st.rerun()
    st.divider()
    st.caption("실행 모드")
    if torch.cuda.is_available():
        st.success(f"Deep AI · {torch.cuda.get_device_name(0)}")
    else:
        st.warning("Deep AI · CPU")
    st.info("ROS2 미설치 환경에서는 동일 이벤트 계약을 사용하는 Mock Runtime으로 동작합니다.")

if run_clicked:
    with st.spinner("Deep PatchCore 검사와 자동 선별을 실행하는 중입니다..."):
        prediction = load_model(str(MODEL_PATH), MODEL_PATH.stat().st_mtime_ns).predict(Image.open(selected.path).convert("RGB"))
        inspection = PipelineInspection(
            product_id=product_id,
            image_path=str(selected.path),
            defect_type=selected.defect_type,
            is_defect=prediction.is_defect,
            anomaly_score=prediction.anomaly_score,
            raw_score=prediction.raw_score,
            threshold=prediction.threshold,
            latency_ms=prediction.latency_ms,
        )
        run = simulate_sort_pipeline(inspection)
        st.session_state.ros2_latest = {"run": run, "prediction": prediction}
        history = st.session_state.setdefault("ros2_history", [])
        history.append(
            {
                "product_id": product_id,
                "defect_type": selected.defect_type,
                "decision": run.decision,
                "final_state": run.final_state,
                "score": prediction.raw_score,
                "latency_ms": prediction.latency_ms,
            }
        )

latest = st.session_state.get("ros2_latest")
history = st.session_state.get("ros2_history", [])

if latest is None:
    preview_left, preview_right = st.columns([0.55, 0.45], gap="large")
    with preview_left:
        st.subheader("대기 중인 제품")
        st.image(selected.path, width="stretch")
    with preview_right:
        st.subheader("실행하면 보이는 것")
        st.markdown(
            """
            1. 카메라 노드의 제품 이미지 발행
            2. GPU Deep PatchCore 이상 탐지와 히트맵
            3. 판정 노드의 PASS 또는 REJECT 결정
            4. ROS2 Action 피드백과 선별 완료 상태
            5. 토픽·액션 이벤트 로그 및 생산 KPI
            """
        )
    st.stop()

run: PipelineRun = latest["run"]
prediction = latest["prediction"]

step = st.slider("파이프라인 단계 재생", 1, len(run.events), len(run.events))
render_pipeline(run, step)

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("제품", run.inspection.product_id)
c2.metric("AI 판정", run.decision)
c3.metric("Anomaly", f"{run.inspection.raw_score:.4f}")
c4.metric("Threshold", f"{run.inspection.threshold:.4f}")
c5.metric("GPU Latency", f"{run.inspection.latency_ms:.1f} ms")

image_col, heatmap_col, actuator_col = st.columns([1, 1, 0.72], gap="large")
with image_col:
    st.markdown("#### 카메라 입력")
    st.image(prediction.prepared_image, width="stretch")
    st.caption(run.inspection.defect_type)
with heatmap_col:
    st.markdown("#### Deep PatchCore 히트맵")
    st.image(prediction.overlay, width="stretch")
    st.caption(f"{run.decision} · 최종 {run.final_state}")
with actuator_col:
    st.markdown("#### Reject Action")
    visible_feedback = [
        event for event in run.events if event.step <= step and event.interface == "ACTION_FEEDBACK"
    ]
    progress = int(visible_feedback[-1].detail.split("%", 1)[0]) if visible_feedback else 0
    st.progress(progress / 100, text=f"액추에이터 진행률 {progress}%")
    if run.decision == "PASS":
        st.success("PASS LANE으로 통과")
    elif progress == 100:
        st.error("REJECT BIN으로 선별 완료")
    else:
        st.warning("Reject Action 실행 중")
    st.caption("Goal → Feedback 25/50/75/100% → Result")

st.subheader("ROS2 Topic / Action 이벤트")
event_frame = pd.DataFrame(run.event_rows(through_step=step))
st.dataframe(event_frame, hide_index=True, width="stretch")

st.subheader("누적 생산 KPI")
history_frame = pd.DataFrame(history)
total = len(history_frame)
rejects = int((history_frame["decision"] == "REJECT").sum())
k1, k2, k3, k4 = st.columns(4)
k1.metric("총 검사", total)
k2.metric("PASS", total - rejects)
k3.metric("REJECT", rejects)
k4.metric("평균 추론", f"{history_frame['latency_ms'].mean():.1f} ms")
st.dataframe(history_frame.iloc[::-1], hide_index=True, width="stretch")
