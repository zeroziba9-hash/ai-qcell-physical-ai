from __future__ import annotations

from html import escape
import json
import os
from pathlib import Path

import pandas as pd
import streamlit as st

from qcell.auth import MANAGE_CAPA, session_principal
from qcell.traceability import CAPA_TRANSITIONS, TraceabilityStore, trace_events_to_csv
from qcell.ui import inject_global_css, page_header, section_header, status_strip, workflow_strip


ROOT = Path(__file__).resolve().parents[1]
DATABASE_PATH = ROOT / "artifacts" / "traceability" / "qcell.db"


st.set_page_config(page_title="제품 추적성 · AI-QCell", page_icon="🧬", layout="wide")
inject_global_css()
st.markdown(
    """
    <style>
    .trace-capa-grid {display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:10px;margin:.5rem 0 1rem;}
    .trace-capa-card {padding:14px;border:1px solid #273240;border-radius:10px;background:#0f141c;}
    .trace-capa-card[data-severity="CRITICAL"] {box-shadow:inset 2px 0 0 #ff7687;}
    .trace-capa-card[data-severity="WARNING"] {box-shadow:inset 2px 0 0 #f5bd62;}
    .trace-capa-code {color:#59697c;font:700 .56rem/1.2 Consolas,monospace;letter-spacing:.1em;}
    .trace-capa-card h3 {margin:.5rem 0 .25rem;font-size:.84rem;}
    .trace-capa-card p {margin:0;color:#7f8fa1;font-size:.7rem;line-height:1.5;}
    .trace-capa-meta {margin-top:.65rem;color:#9aabba;font:650 .6rem/1.4 Consolas,monospace;}
    </style>
    """,
    unsafe_allow_html=True,
)

store = TraceabilityStore(DATABASE_PATH)
if os.getenv("QCELL_AUTH_MODE", "demo").strip().lower() == "demo":
    store.seed_demo()
principal = session_principal(st.session_state)
can_manage_capa = principal.can(MANAGE_CAPA)
metrics = store.metrics()

page_header(
    "GENEALOGY · CAPA · AUDIT TRAIL",
    "제품 추적성 관제센터",
    "제품 ID 하나로 AI 검사, ROS2 선별, 작업자 검수와 CAPA 조치 이력을 끝까지 추적합니다.",
    status="TRACEABILITY LEDGER ONLINE",
)
status_strip(
    [
        {"label": "Ledger", "value": "SQLITE / WAL", "tone": "good"},
        {"label": "Identity", "value": principal.username, "tone": "good"},
        {
            "label": "CAPA control",
            "value": "WRITE" if can_manage_capa else "READ ONLY",
            "tone": "good" if can_manage_capa else "warn",
        },
        {"label": "Schema", "value": f"V{store.schema_version}", "tone": "good"},
    ]
)
workflow_strip(["제품 등록", "AI 검사", "ROS2 선별", "작업자 검수", "CAPA 종결"])

section_header(
    "추적성 KPI",
    "영구 저장된 제품 계보, REJECT 판정, 미종결 CAPA와 감사 이벤트를 요약합니다.",
    code="LEDGER SCORE / 01",
)
m1, m2, m3, m4 = st.columns(4)
m1.metric("추적 제품", f"{metrics['products']:,}")
m2.metric("REJECT 제품", f"{metrics['rejected']:,}")
m3.metric("미종결 CAPA", f"{metrics['open_capa']:,}")
m4.metric("감사 이벤트", f"{metrics['audit_entries']:,}")

product_tab, capa_tab, audit_tab = st.tabs(["제품 계보", "CAPA 조치", "감사 로그"])

with product_tab:
    section_header(
        "제품 검색",
        "제품 ID, Lot과 이력 상태를 조합해 생산 이력을 조회합니다.",
        code="PRODUCT SEARCH / 02",
    )
    search_a, search_b, search_c = st.columns([1.2, 0.9, 0.9])
    with search_a:
        product_query = st.text_input("제품 ID", placeholder="TRACE-00034")
    with search_b:
        lots = ["전체", *store.lots()]
        lot_filter = st.selectbox("Lot", lots)
    with search_c:
        status_filter = st.selectbox(
            "이력 상태", ["전체", "PASS_THROUGH", "REJECT", "PENDING"]
        )
    products = store.products(
        query=product_query,
        lot_id="" if lot_filter == "전체" else lot_filter,
        status="" if status_filter == "전체" else status_filter,
    )
    if not products:
        st.info("검색 조건에 해당하는 제품이 없습니다.")
    else:
        st.dataframe(
            pd.DataFrame(product.to_dict() for product in products),
            hide_index=True,
            width="stretch",
            height=300,
        )
        selected_product = st.selectbox(
            "계보 상세 제품",
            [product.product_id for product in products],
            format_func=lambda product_id: next(
                f"{product_id} · {product.lot_id} · {product.latest_status}"
                for product in products
                if product.product_id == product_id
            ),
        )
        timeline = store.timeline(selected_product)
        stage_order = []
        for event in timeline:
            if event.stage not in stage_order:
                stage_order.append(event.stage)
        if stage_order:
            workflow_strip(stage_order)
        timeline_rows = []
        for event in timeline:
            timeline_rows.append(
                {
                    "occurred_at": event.occurred_at,
                    "stage": event.stage,
                    "event_type": event.event_type,
                    "status": event.status,
                    "model_version": event.model_version,
                    "actor": event.actor,
                    "payload": json.dumps(event.payload, ensure_ascii=False),
                }
            )
        st.dataframe(pd.DataFrame(timeline_rows), hide_index=True, width="stretch")
        export_a, export_b = st.columns(2)
        with export_a:
            st.download_button(
                "제품 계보 CSV",
                trace_events_to_csv(timeline).encode("utf-8-sig"),
                file_name=f"{selected_product}_genealogy.csv",
                mime="text/csv",
                width="stretch",
            )
        with export_b:
            st.download_button(
                "제품 계보 JSON",
                json.dumps(
                    [event.to_dict() for event in timeline],
                    ensure_ascii=False,
                    indent=2,
                ).encode("utf-8"),
                file_name=f"{selected_product}_genealogy.json",
                mime="application/json",
                width="stretch",
            )

with capa_tab:
    section_header(
        "CAPA 현황",
        "품질 알람을 확인하고 원인 분석, 시정 조치, 효과 검증과 종결까지 통제합니다.",
        code="CORRECTIVE ACTION / 03",
    )
    capa_filter = st.segmented_control(
        "상태 필터",
        ["ALL", "OPEN", "ACKNOWLEDGED", "IN_PROGRESS", "VERIFIED", "CLOSED"],
        default="ALL",
    )
    cases = store.capa_cases(status="" if capa_filter in {None, "ALL"} else capa_filter)
    if cases:
        cards = []
        for case in cases[:6]:
            cards.append(
                f'<div class="trace-capa-card" data-severity="{escape(case.severity, quote=True)}">'
                f'<div class="trace-capa-code">{escape(case.case_id)} · {escape(case.status)}</div>'
                f'<h3>{escape(case.title)}</h3><p>{escape(case.description)}</p>'
                f'<div class="trace-capa-meta">LOT {escape(case.lot_id or "-")} · OWNER {escape(case.owner or "UNASSIGNED")}</div>'
                "</div>"
            )
        st.markdown(
            f'<div class="trace-capa-grid">{"".join(cards)}</div>',
            unsafe_allow_html=True,
        )
        selected_case_id = st.selectbox(
            "조치 대상 CAPA",
            [case.case_id for case in cases],
            format_func=lambda case_id: next(
                f"{case.status} · {case.title} · {case_id}"
                for case in cases
                if case.case_id == case_id
            ),
        )
        selected_case = store.get_capa(selected_case_id)
        edit_a, edit_b = st.columns(2, gap="large")
        with edit_a:
            owner = st.text_input("담당자", value=selected_case.owner)
            root_cause = st.text_area("근본 원인", value=selected_case.root_cause, height=110)
        with edit_b:
            corrective_action = st.text_area(
                "시정·예방 조치", value=selected_case.corrective_action, height=110
            )
            note = st.text_input("상태 변경 메모")
        next_statuses = sorted(CAPA_TRANSITIONS[selected_case.status])
        if next_statuses:
            next_status = st.selectbox("다음 상태", next_statuses)
            if st.button(
                "CAPA 상태 전환",
                type="primary",
                disabled=not can_manage_capa,
                width="stretch",
            ):
                try:
                    store.transition_capa(
                        selected_case.case_id,
                        next_status,
                        actor=principal.username,
                        owner=owner,
                        root_cause=root_cause,
                        corrective_action=corrective_action,
                        note=note,
                    )
                    st.success(f"{selected_case.case_id} · {next_status} 전환 완료")
                    st.rerun()
                except ValueError as error:
                    st.warning(str(error))
        else:
            st.success("종결된 CAPA입니다.")
        if not can_manage_capa:
            st.info("CAPA 변경은 Quality Manager 또는 Admin 로그인이 필요합니다.")
    else:
        st.info("선택한 상태의 CAPA가 없습니다.")

    with st.expander("수동 CAPA 등록"):
        with st.form("manual-capa"):
            manual_a, manual_b = st.columns(2)
            with manual_a:
                alert_code = st.text_input("알람 코드", value="MANUAL_QUALITY_ISSUE")
                severity = st.selectbox("심각도", ["INFO", "WARNING", "CRITICAL"])
                capa_lot = st.text_input("관련 Lot")
            with manual_b:
                title = st.text_input("CAPA 제목")
                owner_new = st.text_input("초기 담당자")
                description = st.text_area("문제 설명")
            create_submitted = st.form_submit_button(
                "CAPA 등록", type="primary", disabled=not can_manage_capa, width="stretch"
            )
        if create_submitted:
            if not title.strip() or not description.strip():
                st.warning("제목과 조치 계획을 모두 입력해 주세요.")
            else:
                case, created = store.create_capa(
                    alert_code=alert_code,
                    severity=severity,
                    title=title,
                    description=description,
                    actor=principal.username,
                    lot_id=capa_lot,
                    owner=owner_new,
                    dedupe_key=f"manual:{alert_code}:{capa_lot}:{title}",
                )
                if created:
                    st.success(f"{case.case_id}가 등록됐습니다.")
                    st.rerun()
                else:
                    st.info(f"동일한 CAPA {case.case_id}가 이미 존재합니다.")

with audit_tab:
    section_header(
        "변경 감사 로그",
        "제품 이벤트 기록과 CAPA 상태 변경의 행위자, 대상과 상세 정보를 보존합니다.",
        code="IMMUTABLE LOG / 04",
    )
    audit_entries = store.audit_entries(limit=500)
    audit_frame = pd.DataFrame(
        {
            **entry,
            "details": json.dumps(entry["details"], ensure_ascii=False),
        }
        for entry in audit_entries
    )
    st.dataframe(audit_frame, hide_index=True, width="stretch", height=480)
    st.download_button(
        "감사 로그 JSON",
        json.dumps(audit_entries, ensure_ascii=False, indent=2).encode("utf-8"),
        file_name="ai_qcell_audit_log.json",
        mime="application/json",
        width="stretch",
    )
