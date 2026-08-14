from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from qcell import LineSimulator
from qcell.ui import inject_global_css, module_card, page_header, section_header, status_strip, workflow_strip


st.set_page_config(page_title="AI-QCell", page_icon="🏭", layout="wide")
inject_global_css()

page_header(
    "PHYSICAL AI · QUALITY OPERATING SYSTEM",
    "AI-QCell 스마트 품질 셀",
    "비전 검사부터 이상 판정, ROS2 자동 선별, 모델 학습과 엣지 배포까지 하나의 운영 화면에서 연결합니다.",
    status="CONTROL PLANE ONLINE",
)

if "events" not in st.session_state:
    st.session_state.events = []
if "simulator" not in st.session_state:
    st.session_state.simulator = LineSimulator(defect_rate=0.12, seed=7)

events = pd.DataFrame(st.session_state.events)
status_strip(
    [
        {"label": "Vision model", "value": "Deep PatchCore", "tone": "good"},
        {"label": "Decision loop", "value": "PASS / REJECT", "tone": "good"},
        {"label": "Motion bus", "value": "ROS2 Action", "tone": "good"},
        {"label": "Line state", "value": "IDLE" if events.empty else "INSPECTING", "tone": "warn" if events.empty else "good"},
    ]
)
workflow_strip(["제품 투입", "AI 이상 탐지", "품질 판정", "액추에이터 선별"])

with st.sidebar:
    st.markdown("### 생산라인 제어")
    st.caption("Synthetic line simulator · Station 01")
    defect_rate = st.slider("가상 불량률", 0, 50, 12, 1) / 100
    batch_size = st.slider("검사할 제품 수", 1, 50, 10)

    if st.button("검사 배치 실행", type="primary", use_container_width=True):
        simulator = LineSimulator(defect_rate=defect_rate)
        st.session_state.events.extend(simulator.inspect_next().to_dict() for _ in range(batch_size))
        st.rerun()

    if st.button("기록 초기화", use_container_width=True):
        st.session_state.events = []
        st.rerun()

total = len(events)
defects = int((events["result"] == "DEFECT").sum()) if not events.empty else 0
passes = total - defects
defect_rate_actual = defects / total * 100 if total else 0.0
avg_latency = float(events["latency_ms"].mean()) if not events.empty else 0.0

section_header("생산 운영 현황", "검사량, 선별 결과와 추론 지연을 한눈에 확인합니다.", code="LIVE TELEMETRY / 01")
c1, c2, c3, c4 = st.columns(4)
c1.metric("총 검사", f"{total:,}", help="현재 세션에서 처리한 전체 제품 수")
c2.metric("정상 통과", f"{passes:,}", delta="PASS" if passes else None)
c3.metric("불량 제거", f"{defects:,}", delta=f"{defect_rate_actual:.1f}%")
c4.metric("평균 지연시간", f"{avg_latency:.1f} ms", delta="Inference")

if events.empty:
    st.info("왼쪽 제어 패널에서 ‘검사 배치 실행’을 누르면 실시간 생산 로그와 결함 분포가 채워집니다.")
else:
    left, right = st.columns([1.18, 0.82], gap="large")
    with left:
        st.markdown("#### 실시간 검사 로그")
        display = events.sort_index(ascending=False).copy()
        display["confidence"] = display["confidence"].map(lambda value: f"{value:.1%}")
        st.dataframe(display, use_container_width=True, hide_index=True, height=390)

    with right:
        st.markdown("#### 결함 유형 분포")
        defect_events = events[events["result"] == "DEFECT"]
        if defect_events.empty:
            st.success("현재까지 탐지된 불량이 없습니다.")
        else:
            counts = defect_events["defect_type"].value_counts().rename_axis("defect_type").reset_index(name="count")
            figure = px.bar(
                counts,
                x="defect_type",
                y="count",
                color="defect_type",
                text_auto=True,
                color_discrete_sequence=["#55ddff", "#6e8cff", "#ff6b7f"],
            )
            figure.update_layout(
                showlegend=False,
                xaxis_title="결함 유형",
                yaxis_title="건수",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(7,14,26,0.55)",
                font_color="#9fb0c6",
                margin=dict(l=20, r=20, t=20, b=20),
            )
            st.plotly_chart(figure, use_container_width=True)

    latest = events.iloc[-1]
    section_header("최근 액추에이터 명령", code="MOTION RESULT / 02")
    if latest["action"] == "REJECT":
        st.error(f"{latest['product_id']} · REJECT · {latest['defect_type']} 불량품 제거")
    else:
        st.success(f"{latest['product_id']} · PASS_THROUGH · 정상 제품 통과")

section_header(
    "모듈 런치패드",
    "데이터 준비부터 현장 추론과 제어까지 필요한 작업 화면으로 바로 이동합니다.",
    code="APPLICATION MAP / 03",
)
modules = [
    ("01 · CLASSIC VISION", "기준 영상 검사", "기준 제품과 픽셀 차이를 비교하는 빠른 MVP 검사입니다.", "pages/1_vision_inspection.py"),
    ("02 · PATCH MEMORY", "학습형 패치 검사", "정상 제품만으로 학습한 경량 비지도 이상 탐지입니다.", "pages/2_trained_patch_model.py"),
    ("03 · DEEP MODEL", "Deep PatchCore", "MVTec AD 기반 생산 모델과 결함 히트맵을 확인합니다.", "pages/3_deep_patchcore_mvtec.py"),
    ("04 · CLOSED LOOP", "ROS2 자동 선별", "판정에서 액추에이터 피드백까지 폐루프를 재현합니다.", "pages/4_ros2_sorting_pipeline.py"),
    ("05 · LIVE VISION", "실시간 검사", "카메라와 영상 스트림을 프레임 단위로 검사합니다.", "pages/5_realtime_inspection.py"),
    ("06 · DIGITAL TWIN", "액추에이터 트윈", "컨베이어와 선별 게이트 동작을 시각적으로 검증합니다.", "pages/6_actuator_digital_twin.py"),
    ("07 · DATA OPS", "Dataset Studio", "수집, 라벨링과 데이터 분할을 관리합니다.", "pages/7_dataset_studio.py"),
    ("08 · TRAINING", "Training Lab", "학습, 임계값 보정과 평가 리포트를 생성합니다.", "pages/8_training_lab.py"),
    ("09 · MLOPS", "Model Registry", "후보 모델 비교, 배포와 롤백을 제어합니다.", "pages/9_model_registry.py"),
    ("10 · HUMAN LOOP", "Review Queue", "애매한 판정을 검수해 데이터셋으로 되돌립니다.", "pages/10_review_queue.py"),
    ("11 · EDGE AI", "Edge Runtime", "PyTorch, ONNX Runtime과 TensorRT 성능을 비교합니다.", "pages/11_edge_runtime_benchmark.py"),
]

for row_start in range(0, len(modules), 4):
    columns = st.columns(4, gap="medium")
    for column, (code, title, description, page) in zip(columns, modules[row_start : row_start + 4]):
        with column:
            module_card(code, title, description)
            st.page_link(page, label="모듈 열기 →", use_container_width=True)
