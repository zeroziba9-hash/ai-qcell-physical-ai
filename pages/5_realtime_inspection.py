from __future__ import annotations

from pathlib import Path
import re

import pandas as pd
from PIL import Image
import streamlit as st
import torch
from streamlit_webrtc import WebRtcMode, webrtc_streamer

from qcell.deep_patchcore import DeepPatchCore
from qcell.realtime import (
    RealtimeInspectionStore,
    RealtimeVideoProcessor,
    analyze_video,
    inspect_frame,
    records_to_csv,
)


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "deep_patchcore_bottle.pt"
ARTIFACTS = ROOT / "artifacts" / "realtime"
DEFECTS = ARTIFACTS / "defects"

st.set_page_config(page_title="AI-QCell Realtime", page_icon="🎥", layout="wide")
st.markdown(
    """
    <style>
    .block-container {padding-top:1.35rem;padding-bottom:3rem;}
    [data-testid="stMetric"] {background:#0f172a;border:1px solid #263244;padding:14px;border-radius:14px;}
    .live-dot {display:inline-block;width:10px;height:10px;border-radius:50%;background:#22c55e;
      box-shadow:0 0 12px #22c55e;margin-right:8px;animation:pulse 1.4s infinite;}
    @keyframes pulse {50%{opacity:.25;}}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_model() -> DeepPatchCore:
    return DeepPatchCore.load(MODEL_PATH)


def safe_name(name: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9_.-]", "_", Path(name).name)
    return stem or "uploaded_video.mp4"


if "realtime_store" not in st.session_state:
    st.session_state.realtime_store = RealtimeInspectionStore()

store: RealtimeInspectionStore = st.session_state.realtime_store

st.title("🎥 실시간 Deep PatchCore 검사")
st.caption("브라우저 웹캠, 카메라 스냅샷 또는 동영상 파일을 GPU로 검사하고 불량 프레임과 CSV 이력을 저장합니다.")

if not MODEL_PATH.exists():
    st.error("Deep PatchCore 모델이 없습니다. 먼저 학습 스크립트를 실행하세요.")
    st.stop()

model = load_model()
source_mode = st.radio(
    "입력 소스",
    ["실시간 웹캠", "카메라 스냅샷", "동영상 파일"],
    horizontal=True,
)

with st.sidebar:
    st.header("검사 설정")
    inspect_every = st.slider("검사 프레임 간격", 1, 60, 12, help="작을수록 자주 추론합니다.")
    save_defects = st.toggle("불량 프레임 자동 저장", value=True)
    st.caption(f"장치: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    st.caption(f"임계값: {model.threshold:.4f}")
    if st.button("실시간 기록 초기화", width="stretch"):
        store.clear()
        st.rerun()


def show_records(records, latest_overlay) -> None:
    total = len(records)
    rejects = sum(record.decision == "REJECT" for record in records)
    average = sum(record.latency_ms for record in records) / total if total else 0.0
    last_decision = records[-1].decision if records else "WAITING"
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("검사 프레임", total)
    c2.metric("최근 판정", last_decision)
    c3.metric("REJECT", rejects, delta=f"{rejects / total * 100:.1f}%" if total else "0.0%")
    c4.metric("평균 추론", f"{average:.1f} ms")
    if latest_overlay is not None:
        st.image(latest_overlay, caption="최근 AI 검사 히트맵", width="stretch")
    if records:
        frame = pd.DataFrame(record.to_dict() for record in reversed(records[-50:]))
        st.dataframe(frame, hide_index=True, width="stretch")
        st.download_button(
            "검사 이력 CSV 다운로드",
            records_to_csv(records).encode("utf-8-sig"),
            file_name="ai_qcell_realtime_log.csv",
            mime="text/csv",
        )


if source_mode == "실시간 웹캠":
    st.markdown('<span class="live-dot"></span>브라우저 카메라 라이브 모드', unsafe_allow_html=True)
    try:
        context = webrtc_streamer(
            key="qcell-realtime-camera",
            mode=WebRtcMode.SENDRECV,
            media_stream_constraints={"video": True, "audio": False},
            video_processor_factory=lambda: RealtimeVideoProcessor(
                model,
                store,
                inspect_every=inspect_every,
                defect_dir=DEFECTS if save_defects else None,
            ),
            async_processing=True,
            video_html_attrs={"autoPlay": True, "controls": False, "muted": True},
        )
    except (AttributeError, RuntimeError):
        context = None
        st.info("WebRTC 카메라는 실제 Streamlit 브라우저 세션에서 활성화됩니다.")
    st.info("START를 누르고 카메라 사용을 허용하세요. 초록색 영상은 PASS, 빨간 히트맵은 REJECT입니다.")

    @st.fragment(run_every=1.0 if context is not None and context.state.playing else None)
    def live_dashboard() -> None:
        records, latest = store.snapshot()
        show_records(records, latest)

    live_dashboard()

elif source_mode == "카메라 스냅샷":
    camera_image = st.camera_input("제품을 카메라 중앙에 놓고 촬영하세요")
    if camera_image:
        target = Image.open(camera_image).convert("RGB")
        record, overlay = inspect_frame(
            target,
            model,
            frame_index=len(store.snapshot()[0]) + 1,
            source="camera_snapshot",
            defect_dir=DEFECTS if save_defects else None,
        )
        store.update(record, overlay)
        show_records(*store.snapshot())

else:
    uploaded = st.file_uploader("검사할 동영상", type=["mp4", "mov", "avi", "mkv"])
    max_seconds = st.slider("최대 분석 구간(초)", 5, 60, 30, 5)
    if uploaded:
        st.video(uploaded)
        if st.button("동영상 Deep PatchCore 분석", type="primary", width="stretch"):
            ARTIFACTS.mkdir(parents=True, exist_ok=True)
            input_path = ARTIFACTS / safe_name(uploaded.name)
            input_path.write_bytes(uploaded.getvalue())
            output_path = ARTIFACTS / f"analyzed_{input_path.stem}.mp4"
            with st.spinner("프레임을 샘플링하고 결함 히트맵 영상을 생성하는 중입니다..."):
                summary = analyze_video(
                    input_path,
                    model,
                    output_path,
                    inspect_every=inspect_every,
                    max_frames=max_seconds * 30,
                    defect_dir=DEFECTS if save_defects else None,
                    on_record=store.update,
                )
            st.session_state.video_summary = summary

    summary = st.session_state.get("video_summary")
    if summary and Path(summary.output_path).exists():
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("읽은 프레임", summary.frames_read)
        c2.metric("AI 검사", summary.frames_analyzed)
        c3.metric("REJECT", summary.rejects)
        c4.metric("처리시간", f"{summary.elapsed_seconds:.1f}s")
        st.video(str(summary.output_path))
        st.download_button(
            "히트맵 분석 영상 다운로드",
            Path(summary.output_path).read_bytes(),
            file_name=Path(summary.output_path).name,
            mime="video/mp4",
        )
        show_records(list(summary.records), store.snapshot()[1])
