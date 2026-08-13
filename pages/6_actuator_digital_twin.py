from __future__ import annotations

from pathlib import Path

from PIL import Image
import streamlit as st

from qcell.deep_patchcore import DeepPatchCore, load_mvtec_bottle
from qcell.digital_twin import twin_frame
from qcell.ros2_pipeline import PipelineInspection, simulate_sort_pipeline


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "deep_patchcore_bottle.pt"
DATASET_PATH = ROOT / "data" / "mvtec-ad" / "bottle"

st.set_page_config(page_title="AI-QCell Digital Twin", page_icon="🏭", layout="wide")
st.markdown(
    """
    <style>
    .block-container {padding-top:1.35rem;padding-bottom:3rem;}
    [data-testid="stMetric"] {background:#0f172a;border:1px solid #263244;padding:14px;border-radius:14px;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_model() -> DeepPatchCore:
    return DeepPatchCore.load(MODEL_PATH)


def twin_html(decision: str, product_id: str, score: float) -> str:
    reject = decision == "REJECT"
    product_animation = "rejectProduct" if reject else "passProduct"
    gate_animation = "rejectGate" if reject else "idleGate"
    result_color = "#fb7185" if reject else "#4ade80"
    destination = "REJECT BIN" if reject else "PASS LANE"
    return f"""
    <div class="qcell-twin">
      <div class="head"><span>LIVE DIGITAL TWIN</span><strong>{product_id}</strong><b>{decision} · {score:.4f}</b></div>
      <div class="cell">
        <div class="camera"><div class="lens"></div><label>CAMERA</label></div>
        <div class="scan"></div>
        <div class="belt"><div class="stripe"></div></div>
        <div class="product"><span>{product_id[-2:]}</span></div>
        <div class="gate"></div>
        <div class="pass-lane">PASS LANE</div>
        <div class="reject-lane"></div>
        <div class="reject-bin">REJECT<br>BIN</div>
        <div class="controller"><i></i><span>ROS2 PLC</span></div>
        <div class="result">DESTINATION · {destination}</div>
      </div>
      <div class="timeline">
        <span>CAPTURE</span><span>AI INSPECTION</span><span>DECISION</span><span>ACTION</span><span>COMPLETE</span>
        <div class="timeline-bar"></div>
      </div>
    </div>
    <style>
    .qcell-twin{{font-family:Arial,sans-serif;color:#e2e8f0;background:#050b16;border:1px solid #1e3a52;border-radius:18px;padding:18px;overflow:hidden}}
    .head{{display:flex;gap:24px;align-items:center;padding:0 4px 14px}} .head span{{color:#67e8f9;letter-spacing:.14em;font-size:12px}}
    .head strong{{flex:1}} .head b{{color:{result_color}}}
    .cell{{height:360px;position:relative;background:linear-gradient(#0b1423,#07101d);border:1px solid #223650;border-radius:14px;overflow:hidden}}
    .belt{{position:absolute;left:2%;right:2%;top:48%;height:78px;background:#25334a;border:7px solid #475569;border-radius:42px;box-shadow:inset 0 0 0 8px #121d2d}}
    .stripe{{height:100%;background:repeating-linear-gradient(90deg,transparent 0 36px,#64748b 37px 43px);animation:belt 1s linear infinite}}
    .product{{position:absolute;z-index:6;top:46%;left:4%;width:54px;height:72px;border-radius:20px 20px 14px 14px;background:linear-gradient(90deg,#93c5fd,#dbeafe 50%,#60a5fa);border:3px solid #e0f2fe;display:flex;align-items:center;justify-content:center;color:#0f172a;font-weight:bold;animation:{product_animation} 7s ease-in-out infinite}}
    .camera{{position:absolute;z-index:5;left:20%;top:28px;width:88px;height:76px;background:#334155;border:3px solid #64748b;border-radius:12px;text-align:center}}
    .camera:after{{content:'';position:absolute;left:39px;top:76px;width:8px;height:95px;background:#64748b}} .camera label{{position:absolute;top:-22px;left:10px;font-size:11px;color:#94a3b8}}
    .lens{{width:38px;height:38px;margin:17px auto;border-radius:50%;background:#020617;border:8px solid #0ea5e9;box-shadow:0 0 17px #0ea5e9}}
    .scan{{position:absolute;z-index:4;left:22.6%;top:104px;width:5px;height:92px;background:#22d3ee;box-shadow:0 0 18px 6px #22d3ee99;animation:scan 1.4s ease-in-out infinite}}
    .gate{{position:absolute;z-index:7;left:62%;top:43%;width:14px;height:110px;background:#f59e0b;border-radius:8px;transform-origin:50% 10%;animation:{gate_animation} 7s ease-in-out infinite}}
    .reject-lane{{position:absolute;left:62%;top:57%;width:240px;height:48px;background:#412032;border:5px solid #7f1d1d;transform:rotate(31deg);transform-origin:left center}}
    .reject-bin{{position:absolute;left:77%;top:79%;width:112px;height:55px;border:4px solid #ef4444;background:#450a0a;color:#fca5a5;font-weight:bold;text-align:center;padding-top:8px}}
    .pass-lane{{position:absolute;right:3%;top:41%;color:#86efac;font-size:13px;font-weight:bold}}
    .controller{{position:absolute;right:4%;top:38px;width:130px;height:80px;border:2px solid #475569;background:#111827;border-radius:10px;text-align:center;padding-top:18px}}
    .controller i{{display:inline-block;width:12px;height:12px;background:#22c55e;border-radius:50%;box-shadow:0 0 12px #22c55e;margin-right:8px}} .controller span{{font-size:12px}}
    .result{{position:absolute;right:4%;bottom:22px;color:{result_color};font-weight:bold;letter-spacing:.08em}}
    .timeline{{position:relative;display:grid;grid-template-columns:repeat(5,1fr);gap:8px;padding:22px 2px 4px;color:#94a3b8;font-size:11px;text-align:center}}
    .timeline-bar{{position:absolute;left:3%;right:3%;top:13px;height:4px;background:linear-gradient(90deg,#22d3ee,#8b5cf6,#f59e0b,{result_color});transform-origin:left;animation:grow 7s linear infinite}}
    @keyframes belt{{to{{transform:translateX(43px)}}}} @keyframes scan{{50%{{opacity:.22}}}} @keyframes grow{{0%{{transform:scaleX(0)}}100%{{transform:scaleX(1)}}}}
    @keyframes passProduct{{0%{{left:4%;top:46%}}100%{{left:92%;top:46%}}}}
    @keyframes rejectProduct{{0%{{left:4%;top:46%}}52%{{left:60%;top:46%}}72%{{left:72%;top:72%}}100%{{left:78%;top:78%}}}}
    @keyframes rejectGate{{0%,42%{{transform:rotate(0)}}52%,76%{{transform:rotate(-55deg)}}90%,100%{{transform:rotate(0)}}}}
    @keyframes idleGate{{0%,100%{{transform:rotate(0)}}}}
    @media(prefers-reduced-motion:reduce){{*{{animation-duration:.001s!important;animation-iteration-count:1!important}}}}
    </style>
    """


st.title("🏭 자동 선별 액추에이터 디지털 트윈")
st.caption("Deep PatchCore 판정을 컨베이어, 선별 게이트, ROS2 Action 상태와 동기화한 가상 생산 셀입니다.")

if not MODEL_PATH.exists() or not DATASET_PATH.exists():
    st.error("모델 또는 MVTec 데이터가 없습니다.")
    st.stop()

_, samples = load_mvtec_bottle(DATASET_PATH)
by_type: dict[str, list] = {}
for sample in samples:
    by_type.setdefault(sample.defect_type, []).append(sample)

with st.sidebar:
    st.header("디지털 제품")
    defect_type = st.selectbox("제품 유형", list(by_type))
    sample_index = st.slider("샘플", 0, len(by_type[defect_type]) - 1, 0)
    run_clicked = st.button("생산 셀 가동", type="primary", width="stretch")
    st.caption("애니메이션은 7초마다 반복됩니다.")

selected = by_type[defect_type][sample_index]
if run_clicked or "twin_result" not in st.session_state:
    prediction = load_model().predict(Image.open(selected.path).convert("RGB"))
    product_id = f"QCELL-{len(st.session_state.get('twin_history', [])) + 1:04d}"
    inspection = PipelineInspection(
        product_id=product_id,
        image_path=str(selected.path),
        defect_type=selected.defect_type,
        is_defect=prediction.is_defect,
        anomaly_score=prediction.anomaly_score,
        raw_score=prediction.raw_score,
        threshold=prediction.threshold,
        latency_ms=prediction.latency_ms,
    )
    run = simulate_sort_pipeline(inspection)
    st.session_state.twin_result = (prediction, run)
    st.session_state.setdefault("twin_history", []).append(run.decision)

prediction, run = st.session_state.twin_result
st.html(twin_html(run.decision, run.inspection.product_id, run.inspection.raw_score))

c1, c2, c3, c4 = st.columns(4)
c1.metric("제품 ID", run.inspection.product_id)
c2.metric("AI 판정", run.decision)
c3.metric("추론시간", f"{run.inspection.latency_ms:.1f} ms")
c4.metric("최종 위치", run.final_state)

left, right = st.columns(2, gap="large")
with left:
    st.image(prediction.overlay, caption="Deep PatchCore 결함 위치", width="stretch")
with right:
    phase = st.slider("동작 상태 정밀 보기", 0, 100, 100) / 100
    frame = twin_frame(run.decision, phase)
    st.markdown(f"#### `{frame.state}`")
    st.progress(frame.actuator_progress / 100, text=f"액추에이터 {frame.actuator_progress:.0f}%")
    st.code(
        f"product_x={frame.product_x:.1f}%\nproduct_y={frame.product_y:.1f}%\n"
        f"gate_angle={frame.gate_angle:.1f}°\naction=/qcell/reject_product"
    )
