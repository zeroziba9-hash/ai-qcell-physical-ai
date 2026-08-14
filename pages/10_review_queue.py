from __future__ import annotations

from io import BytesIO
from pathlib import Path

import pandas as pd
from PIL import Image
import streamlit as st

from qcell.ui import inject_global_css, page_header, workflow_strip

from qcell.active_learning import ensure_baseline_registered
from qcell.dataset_studio import DatasetStudio
from qcell.deep_patchcore import DeepPatchCore
from qcell.model_registry import ModelRegistry
from qcell.review_queue import ReviewQueue, is_uncertain


ROOT = Path(__file__).resolve().parents[1]
ACTIVE_ROOT = ROOT / "artifacts" / "active_learning"
DATASET_ROOT = ACTIVE_ROOT / "dataset"
REGISTRY_ROOT = ACTIVE_ROOT / "model_registry"
REVIEW_ROOT = ACTIVE_ROOT / "review_queue"
REALTIME_REJECTS = ROOT / "artifacts" / "realtime" / "defects"
BASELINE_MODEL = ROOT / "models" / "deep_patchcore_bottle.pt"
BASELINE_METADATA = ROOT / "models" / "deep_patchcore_bottle.json"
BASELINE_REPORT = ROOT / "docs" / "results" / "deep_patchcore_bottle_report.json"

st.set_page_config(page_title="AI-QCell Review Queue", page_icon="🔎", layout="wide")
inject_global_css()
st.markdown(
    """
    <style>
    .block-container {padding-top:1.35rem;padding-bottom:3rem;}
    [data-testid="stMetric"] {background:#0f172a;border:1px solid #263244;padding:14px;border-radius:14px;}
    </style>
    """,
    unsafe_allow_html=True,
)

dataset = DatasetStudio(DATASET_ROOT)
registry = ModelRegistry(REGISTRY_ROOT)
queue = ReviewQueue(REVIEW_ROOT)
if BASELINE_MODEL.exists():
    ensure_baseline_registered(registry, BASELINE_MODEL, BASELINE_METADATA, BASELINE_REPORT)


@st.cache_resource
def load_model(path: str, modified_ns: int) -> DeepPatchCore:
    del modified_ns
    return DeepPatchCore.load(path)


model_path, model_version = registry.resolve_model_path(BASELINE_MODEL)
model = load_model(str(model_path), model_path.stat().st_mtime_ns) if model_path.exists() else None
all_cases = queue.cases()
pending = queue.cases("pending")
confirmed = queue.cases("confirmed")
corrected = queue.cases("corrected")

page_header(
    "HUMAN-IN-THE-LOOP · REVIEW QUEUE",
    "품질 판정 Review Queue",
    "애매한 판정과 실시간 REJECT를 작업자가 확인하고 정답을 Dataset Studio로 되돌립니다.",
    status="REVIEW STATION READY",
)
workflow_strip(["AI 판정 큐", "작업자 검수", "정답 확정", "Dataset 반환"])

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("검토 대기", len(pending))
c2.metric("예측 확인", len(confirmed))
c3.metric("예측 수정", len(corrected))
c4.metric("누적 검토", len(all_cases))
c5.metric("운영 모델", model_version)

add_tab, review_tab, history_tab = st.tabs(["검토 샘플 추가", "작업자 판정", "검토 이력"])

with add_tab:
    upload_area, realtime_area = st.columns(2, gap="large")
    with upload_area:
        st.subheader("AI 판정 후 큐에 추가")
        upload = st.file_uploader(
            "검토할 이미지",
            type=["png", "jpg", "jpeg", "webp"],
            key="review_upload",
        )
        margin = st.slider("불확실 구간", 0.05, 0.5, 0.15, 0.05)
        if upload and model is not None:
            image = Image.open(BytesIO(upload.getvalue())).convert("RGB")
            prediction = model.predict(image)
            uncertain = is_uncertain(prediction.raw_score, prediction.threshold, margin)
            left, right = st.columns(2)
            left.image(image, caption="원본", width="stretch")
            right.image(prediction.overlay, caption="AI 히트맵", width="stretch")
            m1, m2, m3 = st.columns(3)
            m1.metric("예측", "REJECT" if prediction.is_defect else "PASS")
            m2.metric("Score", f"{prediction.raw_score:.4f}")
            m3.metric("Threshold", f"{prediction.threshold:.4f}")
            if uncertain:
                st.warning("임계값과 가까운 불확실 샘플입니다. 우선 검토를 권장합니다.")
            if st.button("Review Queue에 추가", type="primary", width="stretch"):
                case = queue.add_case(
                    image,
                    predicted_label="defect" if prediction.is_defect else "normal",
                    raw_score=prediction.raw_score,
                    threshold=prediction.threshold,
                    source=f"upload:{upload.name}",
                    model_version=model_version,
                )
                st.success(f"검토 케이스 {case.case_id}를 추가했습니다.")
                st.rerun()
        elif upload and model is None:
            st.error("검사 모델을 찾을 수 없습니다.")

    with realtime_area:
        st.subheader("실시간 REJECT 가져오기")
        st.info(
            "Realtime Inspection이 자동 저장한 불량 프레임을 중복 없이 Review Queue로 가져옵니다. "
            "작업자가 오검출 여부를 확인한 뒤 재학습 데이터로 전환할 수 있습니다."
        )
        realtime_count = len(
            [
                path
                for path in REALTIME_REJECTS.glob("*")
                if path.suffix.lower() in {".png", ".jpg", ".jpeg"}
            ]
        ) if REALTIME_REJECTS.exists() else 0
        st.metric("저장된 REJECT 프레임", realtime_count)
        if st.button(
            "실시간 REJECT 동기화",
            disabled=not REALTIME_REJECTS.exists(),
            width="stretch",
        ):
            added = queue.import_realtime_rejects(REALTIME_REJECTS)
            st.success(f"새 검토 케이스 {added}건을 가져왔습니다.")
            st.rerun()

with review_tab:
    if not pending:
        st.info("검토 대기 샘플이 없습니다.")
    else:
        selected_id = st.selectbox(
            "검토 케이스",
            [case.case_id for case in pending],
            format_func=lambda case_id: next(
                f"{case.predicted_label.upper()} · {case.source} · {case_id[:8]}"
                for case in pending
                if case.case_id == case_id
            ),
        )
        case = next(case for case in pending if case.case_id == selected_id)
        image_column, decision_column = st.columns([0.58, 0.42], gap="large")
        with image_column:
            st.image(case.path(REVIEW_ROOT), width="stretch")
            st.caption(f"모델 {case.model_version} · 불확실도 {case.uncertainty_percent:.1f}%")
        with decision_column:
            st.markdown("#### 작업자 Ground Truth")
            actual_label = st.radio(
                "실제 판정",
                ["normal", "defect"],
                format_func=lambda value: "정상 · PASS" if value == "normal" else "불량 · REJECT",
                horizontal=True,
            )
            defect_type = st.text_input(
                "결함 유형",
                placeholder="예: scratch",
                disabled=actual_label != "defect",
            )
            st.write(f"AI 예측: **{case.predicted_label.upper()}**")
            st.write(f"Raw score: **{case.raw_score:.4f}** / threshold **{case.threshold:.4f}**")
            if st.button("판정 확정 및 Dataset Studio 전송", type="primary", width="stretch"):
                resolved, record = queue.resolve(
                    case.case_id,
                    actual_label=actual_label,
                    dataset=dataset,
                    defect_type=defect_type,
                )
                st.success(f"{resolved.status.upper()} · 데이터셋 레코드 {record.record_id} 생성")
                st.rerun()

with history_tab:
    resolved_cases = [case for case in all_cases if case.status != "pending"]
    if not resolved_cases:
        st.info("완료된 검토 이력이 없습니다.")
    else:
        st.dataframe(
            pd.DataFrame(
                {
                    "케이스": case.case_id,
                    "상태": case.status,
                    "AI 예측": case.predicted_label,
                    "정답": case.actual_label,
                    "결함 유형": case.defect_type or "-",
                    "모델": case.model_version,
                    "Dataset Record": case.dataset_record_id,
                    "완료 시각": case.resolved_at,
                }
                for case in resolved_cases
            ),
            hide_index=True,
            width="stretch",
        )
