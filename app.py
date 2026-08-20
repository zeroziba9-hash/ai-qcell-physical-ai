from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from qcell import LineSimulator
from qcell.auth import RUN_INSPECTION, session_principal
from qcell.traceability import TraceabilityStore
from qcell.ui import inject_global_css, module_grid, page_header, section_header, status_strip, workflow_strip


ROOT = Path(__file__).resolve().parent
TRACE_DATABASE = ROOT / "artifacts" / "traceability" / "qcell.db"

st.set_page_config(page_title="AI-QCell", page_icon="🏭", layout="wide")
inject_global_css()
principal = session_principal(st.session_state)
trace_store = TraceabilityStore(TRACE_DATABASE)

page_header(
    "PHYSICAL AI · QUALITY OPERATING SYSTEM",
    "AI-QCell 스마트 품질 셀",
    "비전 검사부터 이상 판정, ROS2 자동 선별, 모델 학습과 엣지 배포까지 하나의 운영 화면에서 연결합니다.",
    status="CELL OS / RELEASE 2.0",
)

if "events" not in st.session_state:
    st.session_state.events = []
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

    if st.button(
        "검사 배치 실행",
        type="primary",
        disabled=not principal.can(RUN_INSPECTION),
        use_container_width=True,
    ):
        simulator = LineSimulator(defect_rate=defect_rate, seed=7 + len(st.session_state.events))
        lot_id = f"LIVE-{datetime.now().strftime('%Y%m%d')}"
        for _ in range(batch_size):
            event = simulator.inspect_next().to_dict()
            event["product_id"] = f"Q-{len(st.session_state.events) + 1:05d}"
            event["lot_id"] = lot_id
            st.session_state.events.append(event)
            trace_store.record_inspection(
                event,
                model_version="deep-patchcore-production",
                actor=principal.username,
            )
        st.rerun()
    if not principal.can(RUN_INSPECTION):
        st.caption("배치 실행은 Operator 이상 로그인이 필요합니다.")

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
                color_discrete_sequence=["#5be0b8", "#6aa7ff", "#ff7687"],
            )
            figure.update_layout(
                showlegend=False,
                xaxis_title="결함 유형",
                yaxis_title="건수",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(12,17,24,0.8)",
                font_color="#8796a8",
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
    "운영 모듈",
    "현장 검사와 자동 선별을 중심으로 데이터·모델 운영 도구까지 연결합니다.",
    code="APPLICATION MAP / 03",
)
module_grid(
    [
        {
            "code": "03 / PRODUCTION VISION",
            "title": "Deep PatchCore 산업 결함검사",
            "description": "MVTec AD 기반 생산 모델과 패치 단위 이상 위치를 실시간으로 분석합니다.",
            "tag": "PRIMARY INSPECTION",
            "href": "/deep_patchcore_mvtec",
            "featured": True,
        },
        {
            "code": "05 / CLOSED-LOOP CONTROL",
            "title": "ROS2 자동 선별 파이프라인",
            "description": "검사 판정부터 Reject Action 피드백까지 Physical AI 폐루프를 재현합니다.",
            "tag": "PRIMARY AUTOMATION",
            "href": "/ros2_sorting_pipeline",
            "featured": True,
        },
        {"code": "04 / LIVE VISION", "title": "실시간 검사", "description": "카메라와 영상 스트림을 프레임 단위로 검사합니다.", "href": "/realtime_inspection"},
        {"code": "06 / DIGITAL TWIN", "title": "액추에이터 트윈", "description": "컨베이어와 선별 게이트 동작을 시각적으로 검증합니다.", "href": "/actuator_digital_twin"},
        {"code": "07 / EDGE AI", "title": "Edge Runtime", "description": "PyTorch, ONNX Runtime과 TensorRT 성능을 비교합니다.", "href": "/edge_runtime_benchmark"},
        {"code": "01 / CLASSIC VISION", "title": "기준 영상 검사", "description": "기준 제품과 픽셀 차이를 비교하는 빠른 MVP 검사입니다.", "href": "/vision_inspection"},
        {"code": "02 / PATCH MEMORY", "title": "학습형 패치 검사", "description": "정상 제품만으로 학습한 경량 비지도 이상 탐지입니다.", "href": "/trained_patch_model"},
        {"code": "08 / DATA OPS", "title": "Dataset Studio", "description": "수집, 라벨링과 데이터 분할을 관리합니다.", "href": "/dataset_studio"},
        {"code": "09 / TRAINING", "title": "Training Lab", "description": "학습, 임계값 보정과 평가 리포트를 생성합니다.", "href": "/training_lab"},
        {"code": "10 / MLOPS", "title": "Model Registry", "description": "후보 모델 비교, 배포와 롤백을 제어합니다.", "href": "/model_registry"},
        {"code": "11 / HUMAN LOOP", "title": "Review Queue", "description": "애매한 판정을 검수해 데이터셋으로 되돌립니다.", "href": "/review_queue"},
        {"code": "12 / QUALITY OPS", "title": "교대 품질 분석", "description": "SPC 관리도, 결함 Pareto와 운영 알람을 교대 리포트로 저장합니다.", "href": "/quality_analytics"},
        {"code": "14 / TRACEABILITY", "title": "제품 추적성·CAPA", "description": "제품 계보, 시정조치 상태와 변경 감사 로그를 영구 보존합니다.", "href": "/traceability"},
        {"code": "13 / SECURITY", "title": "접근 제어", "description": "검사자·품질관리자·관리자의 운영 권한을 분리합니다.", "href": "/access_control"},
    ]
)
