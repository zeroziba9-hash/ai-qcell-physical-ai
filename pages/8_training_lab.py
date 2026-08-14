from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from qcell.ui import inject_global_css, page_header, workflow_strip
import torch

from qcell.active_learning import (
    TrainingConfig,
    ensure_baseline_registered,
    train_and_register,
)
from qcell.dataset_studio import DatasetStudio
from qcell.model_registry import ModelRegistry


ROOT = Path(__file__).resolve().parents[1]
ACTIVE_ROOT = ROOT / "artifacts" / "active_learning"
DATASET_ROOT = ACTIVE_ROOT / "dataset"
REGISTRY_ROOT = ACTIVE_ROOT / "model_registry"
BASELINE_MODEL = ROOT / "models" / "deep_patchcore_bottle.pt"
BASELINE_METADATA = ROOT / "models" / "deep_patchcore_bottle.json"
BASELINE_REPORT = ROOT / "docs" / "results" / "deep_patchcore_bottle_report.json"

st.set_page_config(page_title="AI-QCell Training Lab", page_icon="🧪", layout="wide")
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
if BASELINE_MODEL.exists():
    ensure_baseline_registered(registry, BASELINE_MODEL, BASELINE_METADATA, BASELINE_REPORT)

bundle = dataset.training_bundle()
stats = dataset.statistics()
device_label = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
ready = len(bundle.train_normal) >= 3 and len(bundle.validation_normal) >= 1

inject_global_css()
page_header(
    "MODEL LIFECYCLE · TRAINING LAB",
    "Deep PatchCore Training Lab",
    "Dataset Studio의 정상 패치로 학습하고 라벨된 검증 데이터로 임계값을 자동 보정합니다.",
    status="TRAINING WORKSPACE READY",
)
workflow_strip(["데이터 로드", "모델 학습", "임계값 보정", "Registry 등록"])

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("정상 TRAIN", len(bundle.train_normal))
c2.metric("정상 VALID", len(bundle.validation_normal))
c3.metric("임계값 보정", len(bundle.calibration))
c4.metric("최종 TEST", len(bundle.evaluation))
c5.metric("학습 장치", device_label)

settings, results = st.columns([0.34, 0.66], gap="large")
with settings:
    st.subheader("학습 설정")
    display_name = st.text_input("모델 이름", value=f"QCell Custom {stats['total']} samples")
    batch_size = st.select_slider("Batch Size", options=[1, 2, 4, 8, 12, 16], value=8)
    candidate_size = st.select_slider(
        "Candidate Patches", options=[500, 1000, 2000, 4000, 8000, 14000], value=4000
    )
    coreset_size = st.select_slider(
        "Memory Bank", options=[64, 128, 256, 512, 900], value=256
    )
    seed = st.number_input("Seed", min_value=0, max_value=99999, value=42)
    auto_deploy = st.toggle("학습 성공 후 즉시 배포", value=False)
    st.caption(
        "불량 이미지는 메모리 뱅크 학습에 사용하지 않고, 정상/불량을 가장 잘 구분하는 "
        "임계값과 평가 지표 계산에만 사용합니다."
    )
    if not ready:
        st.warning("Dataset Studio에서 정상 TRAIN 3장 이상, 정상 VALID 1장 이상을 준비하세요.")
    train_clicked = st.button(
        "Deep PatchCore 재학습 시작",
        type="primary",
        disabled=not ready,
        width="stretch",
    )

with results:
    st.subheader("학습 결과")
    if train_clicked:
        config = TrainingConfig(
            batch_size=int(batch_size),
            candidate_size=int(candidate_size),
            coreset_size=int(coreset_size),
            seed=int(seed),
        )
        with st.status("Deep PatchCore 사용자 모델 학습 중", expanded=True) as status:
            st.write("1/4 · 정상 이미지에서 ResNet18 패치 특징 추출")
            st.write("2/4 · k-center 메모리 뱅크 구성")
            st.write("3/4 · 라벨 검증 데이터로 최적 임계값 탐색")
            st.write("4/4 · 테스트 평가 및 Model Registry 등록")
            try:
                result = train_and_register(dataset, registry, config, display_name)
                if auto_deploy:
                    registry.deploy(result.version.version_id, reason="training-lab-auto-deploy")
                st.session_state.training_version_id = result.version.version_id
                st.session_state.training_score_rows = list(result.score_rows)
                status.update(label="학습과 모델 등록 완료", state="complete")
            except Exception as error:
                status.update(label="학습 실패", state="error")
                st.exception(error)

    version_id = st.session_state.get("training_version_id")
    if version_id:
        try:
            version = registry.get(version_id)
        except KeyError:
            version = None
        if version is not None:
            metrics = version.metrics
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("F1", f"{float(metrics.get('f1', 0)):.3f}")
            m2.metric("AUROC", f"{float(metrics.get('auroc', 0)):.3f}")
            m3.metric("Precision", f"{float(metrics.get('precision', 0)):.3f}")
            m4.metric("Recall", f"{float(metrics.get('recall', 0)):.3f}")
            m5.metric("Threshold", f"{version.threshold:.4f}")
            st.success(f"등록 완료 · {version.version_id} · {version.training_seconds:.1f}초")

            confusion = metrics.get("confusion_matrix", [[0, 0], [0, 0]])
            left_chart, right_chart = st.columns(2)
            with left_chart:
                st.markdown("#### Confusion Matrix")
                st.dataframe(
                    pd.DataFrame(
                        confusion,
                        index=["Actual Normal", "Actual Defect"],
                        columns=["Pred Normal", "Pred Defect"],
                    ),
                    width="stretch",
                )
            with right_chart:
                st.markdown("#### ROC Curve")
                roc = metrics.get("curves", {}).get("roc", [])
                if roc:
                    frame = pd.DataFrame(roc).set_index("fpr")
                    st.line_chart(frame, x_label="False Positive Rate", y_label="True Positive Rate")

            score_rows = st.session_state.get("training_score_rows", [])
            if score_rows:
                st.markdown("#### 테스트 샘플별 판정")
                st.dataframe(pd.DataFrame(score_rows), hide_index=True, width="stretch")
    else:
        versions = registry.versions()
        if versions:
            latest = versions[0]
            st.info(
                f"최근 등록 모델: {latest.display_name} · `{latest.version_id}` · "
                f"F1 {float(latest.metrics.get('f1', 0)):.3f}"
            )
        else:
            st.info("Dataset Studio에서 데이터를 준비한 뒤 첫 사용자 모델을 학습하세요.")
