from __future__ import annotations

from io import BytesIO
from pathlib import Path

import pandas as pd
from PIL import Image
import streamlit as st

from qcell.ui import inject_global_css, page_header, workflow_strip

from qcell.dataset_studio import DatasetStudio


ROOT = Path(__file__).resolve().parents[1]
ACTIVE_ROOT = ROOT / "artifacts" / "active_learning"
DATASET_ROOT = ACTIVE_ROOT / "dataset"
MVTEC_ROOT = ROOT / "data" / "mvtec-ad" / "bottle"

st.set_page_config(page_title="AI-QCell Dataset Studio", page_icon="📸", layout="wide")
st.markdown(
    """
    <style>
    .block-container {padding-top:1.35rem;padding-bottom:3rem;}
    [data-testid="stMetric"] {background:#0f172a;border:1px solid #263244;padding:14px;border-radius:14px;}
    .flow {padding:12px 16px;border:1px solid #0ea5e9;border-radius:12px;background:#082f49;color:#e0f2fe;}
    </style>
    """,
    unsafe_allow_html=True,
)

dataset = DatasetStudio(DATASET_ROOT)
stats = dataset.statistics()

inject_global_css()
page_header(
    "DATA OPERATIONS · DATASET STUDIO",
    "현장 데이터셋 스튜디오",
    "현장 이미지를 수집하고 정상·불량 라벨과 학습·검증·테스트 분할을 관리합니다.",
    status="DATA PIPELINE READY",
)
workflow_strip(["이미지 수집", "품질 라벨", "재현 가능 분할", "학습 준비"])
st.markdown(
    '<div class="flow">수집 → 라벨링 → 데이터 분할 → Deep PatchCore 학습 → 검토 데이터 재유입</div>',
    unsafe_allow_html=True,
)

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("전체 이미지", stats["total"])
c2.metric("정상", stats["by_label"]["normal"])
c3.metric("불량", stats["by_label"]["defect"])
c4.metric("미라벨", stats["by_label"]["unlabeled"])
c5.metric("Dataset ID", dataset.fingerprint()[:8])

collect_tab, split_tab, library_tab = st.tabs(["이미지 수집", "분할 관리", "데이터 라이브러리"])

with collect_tab:
    left, right = st.columns([0.58, 0.42], gap="large")
    with left:
        st.subheader("업로드 수집")
        uploads = st.file_uploader(
            "제품 이미지",
            type=["png", "jpg", "jpeg", "webp"],
            accept_multiple_files=True,
            key="dataset_uploads",
        )
        label = st.radio(
            "라벨",
            ["normal", "defect", "unlabeled"],
            format_func=lambda value: {
                "normal": "정상 · PASS",
                "defect": "불량 · REJECT",
                "unlabeled": "미확인 · REVIEW",
            }[value],
            horizontal=True,
        )
        defect_type = st.text_input(
            "결함 유형",
            placeholder="예: scratch, contamination",
            disabled=label != "defect",
        )
        if st.button("선택 이미지 저장", type="primary", disabled=not uploads, width="stretch"):
            added = 0
            for upload in uploads:
                image = Image.open(BytesIO(upload.getvalue())).convert("RGB")
                dataset.add_image(
                    image,
                    label=label,
                    defect_type=defect_type,
                    source=f"upload:{upload.name}",
                )
                added += 1
            st.success(f"{added}장을 Dataset Studio에 저장했습니다.")
            st.rerun()

        camera = st.camera_input("카메라로 제품 촬영", key="dataset_camera")
        if camera and st.button("촬영 이미지 저장", width="stretch"):
            image = Image.open(BytesIO(camera.getvalue())).convert("RGB")
            dataset.add_image(
                image,
                label=label,
                defect_type=defect_type,
                source="camera:dataset-studio",
            )
            st.success("촬영 이미지를 저장했습니다.")
            st.rerun()

    with right:
        st.subheader("포트폴리오 데모 데이터")
        st.info(
            "로컬 MVTec bottle에서 정상 40장과 불량 12장을 복사해 즉시 학습 가능한 "
            "Dataset Studio 데모를 구성합니다. 원본 데이터는 변경하지 않습니다."
        )
        if not MVTEC_ROOT.exists():
            st.warning("MVTec bottle 데이터가 없습니다.")
        if st.button(
            "MVTec 데모 52장 불러오기",
            disabled=not MVTEC_ROOT.exists(),
            width="stretch",
        ):
            with st.spinner("이미지와 라벨을 Dataset Studio로 가져오는 중..."):
                added = dataset.seed_from_mvtec(MVTEC_ROOT, normal_count=40, defect_count=12)
            st.success(f"새 이미지 {added}장을 추가하고 데이터 분할을 완료했습니다.")
            st.rerun()

with split_tab:
    st.subheader("재현 가능한 데이터 분할")
    col1, col2, col3 = st.columns(3)
    train_ratio = col1.slider("정상 학습 비율", 0.5, 0.9, 0.8, 0.05)
    validation_ratio = col2.slider("정상 검증 비율", 0.05, 0.3, 0.1, 0.05)
    seed = col3.number_input("랜덤 시드", min_value=0, max_value=99999, value=42)
    if train_ratio + validation_ratio >= 1:
        st.error("학습+검증 비율은 1보다 작아야 합니다.")
    elif st.button("전체 데이터 다시 분할", type="primary", width="stretch"):
        dataset.assign_splits(train_ratio, validation_ratio, int(seed))
        st.success("정상은 학습/검증/테스트, 불량은 검증/테스트로 분할했습니다.")
        st.rerun()

    split_stats = dataset.statistics()["by_split"]
    columns = st.columns(5)
    for column, split_name in zip(columns, ["train", "validation", "test", "review", "unassigned"]):
        column.metric(split_name.upper(), split_stats[split_name])
    st.caption("Deep PatchCore의 메모리 뱅크에는 정상 TRAIN 이미지만 사용됩니다. 불량 이미지는 임계값 보정과 평가에만 사용됩니다.")

with library_tab:
    records = dataset.records()
    if not records:
        st.info("아직 저장된 이미지가 없습니다. 업로드하거나 데모 데이터를 불러오세요.")
    else:
        table = pd.DataFrame(
            {
                "ID": record.record_id,
                "라벨": record.label,
                "결함 유형": record.defect_type or "-",
                "분할": record.split,
                "출처": record.source,
                "등록 시각": record.created_at,
            }
            for record in reversed(records)
        )
        st.dataframe(table, hide_index=True, width="stretch", height=320)
        st.markdown("#### 최근 이미지")
        recent = list(reversed(records))[:12]
        for start in range(0, len(recent), 4):
            columns = st.columns(4)
            for column, record in zip(columns, recent[start : start + 4]):
                with column:
                    st.image(record.path(DATASET_ROOT), width="stretch")
                    st.caption(f"{record.label.upper()} · {record.split} · {record.record_id[:6]}")
