from __future__ import annotations

from html import escape
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from qcell.quality_analytics import (
    QualityReportStore,
    build_quality_report,
    generate_demo_shift,
    quality_events_to_csv,
    quality_report_to_json,
)
from qcell.ui import (
    inject_global_css,
    page_header,
    section_header,
    status_strip,
    workflow_strip,
)


ROOT = Path(__file__).resolve().parents[1]
REPORT_ROOT = ROOT / "artifacts" / "quality_reports"


st.set_page_config(page_title="품질 분석 리포트 · AI-QCell", page_icon="📈", layout="wide")
inject_global_css()
st.markdown(
    """
    <style>
    .quality-alert-grid {display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:10px;margin:.4rem 0 1rem;}
    .quality-alert {min-height:108px;padding:14px 15px;border:1px solid #273240;border-radius:10px;background:#0f141c;}
    .quality-alert[data-severity="critical"] {border-color:rgba(255,118,135,.34);box-shadow:inset 2px 0 0 #ff7687;}
    .quality-alert[data-severity="warning"] {border-color:rgba(245,189,98,.28);box-shadow:inset 2px 0 0 #f5bd62;}
    .quality-alert[data-severity="info"] {border-color:rgba(106,167,255,.25);box-shadow:inset 2px 0 0 #6aa7ff;}
    .quality-alert-code {color:#59697c;font:700 .57rem/1.2 Consolas,monospace;letter-spacing:.11em;}
    .quality-alert h3 {margin:.5rem 0 .32rem;font-size:.86rem;}
    .quality-alert p {margin:0;color:#7f8fa1;font-size:.7rem;line-height:1.55;word-break:keep-all;}
    .quality-alert-value {margin-top:.58rem;color:#cbd5df;font:700 .64rem/1.2 Consolas,monospace;}
    .quality-ok {padding:16px;border:1px solid rgba(91,224,184,.2);border-radius:10px;background:rgba(91,224,184,.045);color:#9cebd2;font-size:.78rem;}
    .quality-note {margin:.4rem 0 0;color:#627185;font:600 .63rem/1.5 Consolas,monospace;}
    </style>
    """,
    unsafe_allow_html=True,
)

page_header(
    "QUALITY OPS · SHIFT INTELLIGENCE",
    "교대 품질 분석 센터",
    "검사 이력을 SPC 관리도와 결함 Pareto로 분석하고, 운영 알람과 감사 가능한 교대 리포트를 생성합니다.",
    status="QUALITY ANALYTICS READY",
)

if "quality_demo_events" not in st.session_state:
    st.session_state.quality_demo_events = generate_demo_shift()

dashboard_events = [dict(event) for event in st.session_state.get("events", [])]
source_options = ["교대 시뮬레이션"]
if dashboard_events:
    source_options.append("운영 대시보드 세션")

with st.sidebar:
    st.markdown("### 품질 분석 조건")
    source_label = st.radio("데이터 소스", source_options, horizontal=False)
    sample_count = st.slider("교대 검사 수", 120, 600, 240, 40)
    baseline_rate = st.slider("정상 구간 불량률", 0.0, 10.0, 3.0, 0.5) / 100.0
    inject_drift = st.toggle("후반 공정 드리프트 주입", value=True)
    drift_rate = st.slider(
        "드리프트 구간 불량률",
        5.0,
        30.0,
        24.0,
        1.0,
        disabled=not inject_drift,
    ) / 100.0
    seed = int(st.number_input("재현 시드", min_value=1, max_value=9999, value=23))
    if st.button("새 교대 데이터 생성", type="primary", width="stretch"):
        st.session_state.quality_demo_events = generate_demo_shift(
            sample_count,
            baseline_defect_rate=baseline_rate,
            drift_defect_rate=drift_rate if inject_drift else baseline_rate,
            drift_start_ratio=0.72 if inject_drift else 1.0,
            seed=seed,
        )
        st.rerun()

    st.markdown("### 판정 기준")
    subgroup_size = st.select_slider("SPC 소그룹 크기", options=[10, 20, 30, 40], value=20)
    target_rate = st.slider("목표 불량률", 0.5, 15.0, 5.0, 0.5)
    latency_sla = st.slider("p95 지연 SLA", 20, 100, 50, 5)
    confidence_floor = st.slider("평균 신뢰도 하한", 75, 99, 90, 1)

if source_label == "운영 대시보드 세션":
    source_name = "dashboard-session"
    raw_events = dashboard_events
else:
    source_name = "shift-simulation"
    raw_events = [dict(event) for event in st.session_state.quality_demo_events]

events: list[dict[str, object]] = []
for event in raw_events:
    events.append(
        {
            **event,
            "lot_id": event.get("lot_id", "LIVE-SESSION"),
            "process_phase": event.get("process_phase", "live"),
        }
    )

all_lots = sorted({str(event.get("lot_id", "UNASSIGNED")) for event in events})
all_phases = sorted({str(event.get("process_phase", "live")) for event in events})
section_header(
    "분석 범위",
    "Lot과 공정 구간을 선택하면 모든 KPI, 관리도, 알람과 내보내기 파일에 동일하게 적용됩니다.",
    code="FILTER CONTEXT / 01",
)
filter_a, filter_b = st.columns([1.2, 0.8], gap="large")
with filter_a:
    selected_lots = st.multiselect("Lot 필터", all_lots, default=all_lots)
with filter_b:
    selected_phases = st.multiselect(
        "공정 구간", all_phases, default=all_phases
    )

filtered_events = [
    event
    for event in events
    if str(event.get("lot_id")) in selected_lots
    and str(event.get("process_phase", "live")) in selected_phases
]
if not filtered_events:
    st.warning("선택한 조건에 해당하는 검사 기록이 없습니다.")
    st.stop()

report = build_quality_report(
    filtered_events,
    subgroup_size=subgroup_size,
    target_defect_rate_percent=target_rate,
    latency_sla_ms=float(latency_sla),
    confidence_floor_percent=float(confidence_floor),
    source=source_name,
)
summary = dict(report["summary"])
control_rows = list(report["control_chart"])
pareto_rows = list(report["defect_pareto"])
alerts = list(report["alerts"])
unstable_groups = sum(bool(row["out_of_control"]) for row in control_rows)

status_strip(
    [
        {"label": "Data source", "value": source_name, "tone": "good"},
        {"label": "Active lots", "value": f"{len(selected_lots)} LOT", "tone": "good"},
        {
            "label": "Process state",
            "value": "ACTION REQUIRED" if alerts else "IN CONTROL",
            "tone": "bad" if alerts else "good",
        },
        {
            "label": "SPC signal",
            "value": f"{unstable_groups} OUTLIER",
            "tone": "bad" if unstable_groups else "good",
        },
    ]
)
workflow_strip(["검사 이력 수집", "소그룹 SPC 분석", "결함 Pareto", "교대 리포트 승인"])

section_header(
    "교대 품질 KPI",
    "필터링된 검사 이력의 수율, 불량 PPM, 추론 지연과 활성 알람을 요약합니다.",
    code="SHIFT SCORECARD / 02",
)
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("검사 수", f"{int(summary['inspected']):,}")
k2.metric("직행 수율", f"{float(summary['first_pass_yield_percent']):.2f}%")
k3.metric("불량 PPM", f"{int(summary['defect_ppm']):,}", delta=f"{float(summary['defect_rate_percent']):.2f}%")
k4.metric("p95 지연", f"{float(summary['p95_latency_ms']):.1f} ms", delta=f"SLA {latency_sla} ms")
k5.metric("활성 알람", f"{len(alerts)}", delta="Review" if alerts else "Stable")
st.markdown(
    f'<div class="quality-note">AVG CONFIDENCE {float(summary["average_confidence_percent"]):.2f}% · '
    f'AVG LATENCY {float(summary["average_latency_ms"]):.2f} ms · SUBGROUP {subgroup_size}</div>',
    unsafe_allow_html=True,
)

section_header(
    "SPC 공정 관리도",
    "소그룹별 불량률을 3σ 관리한계와 목표 불량률에 대조해 공정 드리프트를 조기에 탐지합니다.",
    code="P-CHART / 03",
)
control_frame = pd.DataFrame(control_rows)
control_figure = go.Figure()
control_figure.add_trace(
    go.Scatter(
        x=control_frame["subgroup"],
        y=control_frame["upper_control_limit_percent"],
        mode="lines",
        name="UCL 3σ",
        line=dict(color="#ff7687", width=1, dash="dot"),
    )
)
control_figure.add_trace(
    go.Scatter(
        x=control_frame["subgroup"],
        y=control_frame["center_line_percent"],
        mode="lines",
        name="Center line",
        line=dict(color="#6aa7ff", width=1, dash="dash"),
    )
)
control_figure.add_trace(
    go.Scatter(
        x=control_frame["subgroup"],
        y=control_frame["defect_rate_percent"],
        mode="lines+markers",
        name="Defect rate",
        line=dict(color="#5be0b8", width=2),
        marker=dict(
            size=8,
            color=["#ff7687" if value else "#5be0b8" for value in control_frame["out_of_control"]],
        ),
        hovertemplate="Subgroup %{x}<br>Defect %{y:.2f}%<extra></extra>",
    )
)
control_figure.add_hline(
    y=target_rate,
    line_color="#f5bd62",
    line_dash="dash",
    annotation_text=f"Target {target_rate:.1f}%",
    annotation_position="top left",
)
control_figure.update_layout(
    height=365,
    margin=dict(l=20, r=20, t=24, b=20),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(12,17,24,.8)",
    font_color="#8796a8",
    legend=dict(orientation="h", y=1.12, x=0),
    xaxis_title="SPC subgroup",
    yaxis_title="Defect rate (%)",
)
st.plotly_chart(control_figure, width="stretch", config={"displayModeBar": False})

left, right = st.columns([1.08, 0.92], gap="large")
with left:
    section_header(
        "결함 Pareto",
        "누적 기여도가 높은 결함부터 개선 우선순위를 정합니다.",
        code="DEFECT MIX / 04",
    )
    if pareto_rows:
        pareto_frame = pd.DataFrame(pareto_rows)
        pareto_figure = go.Figure()
        pareto_figure.add_trace(
            go.Bar(
                x=pareto_frame["defect_type"],
                y=pareto_frame["count"],
                name="Defects",
                marker_color="#5be0b8",
                hovertemplate="%{x}<br>%{y} defects<extra></extra>",
            )
        )
        pareto_figure.add_trace(
            go.Scatter(
                x=pareto_frame["defect_type"],
                y=pareto_frame["cumulative_percent"],
                name="Cumulative",
                yaxis="y2",
                mode="lines+markers",
                line=dict(color="#f5bd62", width=2),
            )
        )
        pareto_figure.update_layout(
            height=340,
            margin=dict(l=20, r=20, t=24, b=20),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(12,17,24,.8)",
            font_color="#8796a8",
            legend=dict(orientation="h", y=1.12, x=0),
            yaxis=dict(title="Count"),
            yaxis2=dict(title="Cumulative %", overlaying="y", side="right", range=[0, 105]),
        )
        st.plotly_chart(pareto_figure, width="stretch", config={"displayModeBar": False})
    else:
        st.success("선택한 범위에 불량 기록이 없습니다.")

with right:
    section_header(
        "운영 알람",
        "공정·품질·지연 기준을 동시에 평가해 조치가 필요한 항목을 표시합니다.",
        code="RULE ENGINE / 05",
    )
    if alerts:
        alert_cards = []
        for alert in alerts:
            alert_cards.append(
                f'<div class="quality-alert" data-severity="{escape(str(alert["severity"]))}">'
                f'<div class="quality-alert-code">{escape(str(alert["code"]))}</div>'
                f'<h3>{escape(str(alert["title"]))}</h3>'
                f'<p>{escape(str(alert["detail"]))}</p>'
                f'<div class="quality-alert-value">MEASURED {float(alert["measured"]):.2f} / LIMIT {float(alert["threshold"]):.2f}</div>'
                "</div>"
            )
        st.markdown(
            f'<div class="quality-alert-grid">{"".join(alert_cards)}</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="quality-ok">모든 품질·공정·지연 지표가 설정한 관리 범위 안에 있습니다.</div>',
            unsafe_allow_html=True,
        )

section_header(
    "리포트 승인 및 내보내기",
    "현재 분석 조건을 JSON 스냅샷으로 보존하거나 원시 검사 이력과 함께 다운로드합니다.",
    code="AUDIT TRAIL / 06",
)
store = QualityReportStore(REPORT_ROOT)
report_name = st.text_input("리포트 이름", value="2026-08-14 A조 교대 품질 리포트")
action_a, action_b, action_c = st.columns([1, 1, 1], gap="medium")
with action_a:
    if st.button("리포트 스냅샷 저장", type="primary", width="stretch"):
        snapshot = store.save(report, name=report_name)
        st.success(f"저장 완료 · {snapshot['snapshot_id']}")
with action_b:
    st.download_button(
        "검사 이력 CSV",
        quality_events_to_csv(filtered_events).encode("utf-8-sig"),
        file_name="ai_qcell_shift_events.csv",
        mime="text/csv",
        width="stretch",
    )
with action_c:
    st.download_button(
        "품질 리포트 JSON",
        quality_report_to_json(report).encode("utf-8"),
        file_name="ai_qcell_quality_report.json",
        mime="application/json",
        width="stretch",
    )

snapshots = store.list()
if snapshots:
    history_rows = []
    for snapshot in snapshots[:20]:
        saved_summary = dict(dict(snapshot["report"])["summary"])
        history_rows.append(
            {
                "snapshot_id": snapshot["snapshot_id"],
                "name": snapshot["name"],
                "saved_at": snapshot["saved_at"],
                "inspected": saved_summary["inspected"],
                "yield_percent": saved_summary["first_pass_yield_percent"],
                "defect_ppm": saved_summary["defect_ppm"],
            }
        )
    with st.expander(f"저장된 리포트 이력 · {len(snapshots)}건"):
        st.dataframe(pd.DataFrame(history_rows), width="stretch", hide_index=True)

with st.expander(f"검사 이력 상세 · {len(filtered_events):,}건"):
    detail = pd.DataFrame(filtered_events).sort_values("timestamp", ascending=False)
    if "confidence" in detail:
        detail["confidence"] = detail["confidence"].map(lambda value: f"{float(value):.1%}")
    st.dataframe(detail, width="stretch", hide_index=True, height=380)
