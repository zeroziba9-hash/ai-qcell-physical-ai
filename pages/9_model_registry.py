from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from qcell.ui import inject_global_css, page_header, workflow_strip

from qcell.active_learning import ensure_baseline_registered
from qcell.model_registry import ModelRegistry


ROOT = Path(__file__).resolve().parents[1]
ACTIVE_ROOT = ROOT / "artifacts" / "active_learning"
REGISTRY_ROOT = ACTIVE_ROOT / "model_registry"
BASELINE_MODEL = ROOT / "models" / "deep_patchcore_bottle.pt"
BASELINE_METADATA = ROOT / "models" / "deep_patchcore_bottle.json"
BASELINE_REPORT = ROOT / "docs" / "results" / "deep_patchcore_bottle_report.json"

st.set_page_config(page_title="AI-QCell Model Registry", page_icon="📦", layout="wide")
st.markdown(
    """
    <style>
    .block-container {padding-top:1.35rem;padding-bottom:3rem;}
    [data-testid="stMetric"] {background:#0f172a;border:1px solid #263244;padding:14px;border-radius:14px;}
    .production {padding:16px;border:1px solid #22c55e;border-radius:14px;background:#052e16;color:#dcfce7;}
    </style>
    """,
    unsafe_allow_html=True,
)

registry = ModelRegistry(REGISTRY_ROOT)
if BASELINE_MODEL.exists():
    ensure_baseline_registered(registry, BASELINE_MODEL, BASELINE_METADATA, BASELINE_REPORT)

versions = registry.versions()
deployed = registry.deployed()

inject_global_css()
page_header(
    "MLOPS CONTROL PLANE · MODEL REGISTRY",
    "운영 모델 레지스트리",
    "모델 성능을 비교하고 운영 버전을 배포하거나 이전 버전으로 즉시 롤백합니다.",
    status="DEPLOYMENT CONTROL READY",
)
workflow_strip(["후보 비교", "배포 승인", "Production 전환", "즉시 Rollback"])

if deployed:
    st.markdown(
        f'<div class="production"><b>PRODUCTION</b> · {deployed.display_name} · '
        f'<code>{deployed.version_id}</code> · threshold {deployed.threshold:.4f}</div>',
        unsafe_allow_html=True,
    )
else:
    st.warning("현재 배포된 모델이 없습니다.")

if not versions:
    st.info("등록된 모델이 없습니다. Training Lab에서 학습을 실행하세요.")
    st.stop()

summary_rows = []
for version in versions:
    summary_rows.append(
        {
            "상태": "PRODUCTION" if deployed and deployed.version_id == version.version_id else "CANDIDATE",
            "버전": version.version_id,
            "이름": version.display_name,
            "F1": float(version.metrics.get("f1", 0.0)),
            "AUROC": float(version.metrics.get("auroc", 0.0)),
            "Precision": float(version.metrics.get("precision", 0.0)),
            "Recall": float(version.metrics.get("recall", 0.0)),
            "Threshold": version.threshold,
            "학습 시간(초)": version.training_seconds,
            "Dataset": version.dataset_fingerprint,
            "등록 시각": version.created_at,
        }
    )

st.markdown("### 버전 비교")
st.dataframe(pd.DataFrame(summary_rows), hide_index=True, width="stretch")

metric_frame = pd.DataFrame(summary_rows).set_index("버전")
chart_columns = [column for column in ["F1", "AUROC", "Precision", "Recall"] if column in metric_frame]
if chart_columns:
    st.bar_chart(metric_frame[chart_columns])

details, actions = st.columns([0.64, 0.36], gap="large")
with details:
    selected_id = st.selectbox(
        "상세 버전",
        [version.version_id for version in versions],
        format_func=lambda version_id: next(
            version.display_name for version in versions if version.version_id == version_id
        )
        + f" · {version_id}",
    )
    selected = registry.get(selected_id)
    metadata_path = REGISTRY_ROOT / selected.metadata_path
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        with st.expander("학습 메타데이터", expanded=False):
            st.json(metadata)

with actions:
    st.subheader("배포 제어")
    is_current = deployed is not None and deployed.version_id == selected_id
    if st.button(
        "선택 버전을 PRODUCTION 배포",
        type="primary",
        disabled=is_current,
        width="stretch",
    ):
        registry.deploy(selected_id, reason="model-registry-ui")
        st.success(f"{selected_id} 배포 완료")
        st.rerun()
    if st.button("이전 배포로 롤백", width="stretch"):
        try:
            rolled_back = registry.rollback()
            st.success(f"{rolled_back.version_id}로 롤백했습니다.")
            st.rerun()
        except ValueError as error:
            st.warning(str(error))
    st.caption("배포 포인터만 원자적으로 변경하므로 원본 모델 파일과 학습 결과는 보존됩니다.")

history = registry.deployment_history()
st.markdown("### 배포 이력")
if history:
    st.dataframe(pd.DataFrame(history), hide_index=True, width="stretch")
else:
    st.info("배포 이력이 없습니다.")
