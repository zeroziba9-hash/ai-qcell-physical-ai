from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from qcell import LineSimulator


st.set_page_config(page_title="AI-QCell", page_icon="🏭", layout="wide")

st.markdown(
    """
    <style>
    .block-container {padding-top: 1.5rem;}
    [data-testid="stMetric"] {background:#111827; border:1px solid #263244; padding:16px; border-radius:14px;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🏭 AI-QCell")
st.caption("Physical AI 기반 스마트팩토리 불량검사 · 자동 분류 시스템")

if "events" not in st.session_state:
    st.session_state.events = []
if "simulator" not in st.session_state:
    st.session_state.simulator = LineSimulator(defect_rate=0.12, seed=7)

with st.sidebar:
    st.header("생산라인 제어")
    defect_rate = st.slider("가상 불량률", 0, 50, 12, 1) / 100
    batch_size = st.slider("검사할 제품 수", 1, 50, 10)

    if st.button("검사 배치 실행", type="primary", use_container_width=True):
        simulator = LineSimulator(defect_rate=defect_rate)
        st.session_state.events.extend(
            simulator.inspect_next().to_dict() for _ in range(batch_size)
        )

    if st.button("기록 초기화", use_container_width=True):
        st.session_state.events = []

events = pd.DataFrame(st.session_state.events)

if events.empty:
    st.info("왼쪽의 ‘검사 배치 실행’을 눌러 가상 생산라인을 시작하세요.")
    st.stop()

total = len(events)
defects = int((events["result"] == "DEFECT").sum())
passes = total - defects
defect_rate_actual = defects / total * 100
avg_latency = events["latency_ms"].mean()

c1, c2, c3, c4 = st.columns(4)
c1.metric("총 검사", f"{total:,}")
c2.metric("정상 통과", f"{passes:,}")
c3.metric("불량 제거", f"{defects:,}", delta=f"{defect_rate_actual:.1f}%")
c4.metric("평균 지연시간", f"{avg_latency:.1f} ms")

left, right = st.columns([1.15, 0.85])

with left:
    st.subheader("실시간 검사 로그")
    display = events.sort_index(ascending=False).copy()
    display["confidence"] = display["confidence"].map(lambda value: f"{value:.1%}")
    st.dataframe(display, use_container_width=True, hide_index=True, height=390)

with right:
    st.subheader("결함 유형 분포")
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
            color_discrete_sequence=["#22d3ee", "#a78bfa", "#fb7185"],
        )
        figure.update_layout(showlegend=False, xaxis_title="결함 유형", yaxis_title="건수")
        st.plotly_chart(figure, use_container_width=True)

latest = events.iloc[-1]
st.subheader("최근 액추에이터 명령")
if latest["action"] == "REJECT":
    st.error(f"{latest['product_id']} · REJECT · {latest['defect_type']} 불량품 제거")
else:
    st.success(f"{latest['product_id']} · PASS_THROUGH · 정상 제품 통과")

