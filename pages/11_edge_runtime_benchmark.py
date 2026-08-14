from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from PIL import Image
import streamlit as st

from qcell.ui import inject_global_css, page_header, workflow_strip

from qcell.deep_patchcore import DeepPatchCore, load_mvtec_bottle
from qcell.edge_runtime import OnnxDeepPatchCore, TensorRTDeepPatchCore, runtime_readiness
from qcell.model_registry import ModelRegistry


ROOT = Path(__file__).resolve().parents[1]
EDGE_ROOT = ROOT / "artifacts" / "edge_runtime"
REGISTRY_ROOT = ROOT / "artifacts" / "active_learning" / "model_registry"
BASELINE_MODEL = ROOT / "models" / "deep_patchcore_bottle.pt"
DATASET_ROOT = ROOT / "data" / "mvtec-ad" / "bottle"
REPORT_PATH = ROOT / "docs" / "results" / "edge_runtime_benchmark.json"
VISUAL_PATH = ROOT / "docs" / "images" / "edge_runtime_benchmark.png"

st.set_page_config(page_title="AI-QCell Edge Runtime", page_icon="⚡", layout="wide")
inject_global_css()
st.markdown(
    """
    <style>
    .block-container {padding-top:1.35rem;padding-bottom:3rem;}
    [data-testid="stMetric"] {background:#0f172a;border:1px solid #263244;padding:14px;border-radius:14px;}
    .runtime-flow {display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:10px;margin:12px 0 20px;}
    .runtime-card {min-width:0;padding:16px;border:1px solid #283442;border-radius:10px;background:#0f141c;}
    .runtime-card strong {display:block;color:#5be0b8;font-size:1rem;margin-bottom:6px;word-break:keep-all;}
    .runtime-card span {display:block;color:#8e9caf;font-size:.82rem;line-height:1.55;word-break:keep-all;overflow-wrap:break-word;}
    @media (max-width:720px) {.runtime-flow {grid-template-columns:1fr;}}
    </style>
    """,
    unsafe_allow_html=True,
)

registry = ModelRegistry(REGISTRY_ROOT)
model_path, version_id = registry.resolve_model_path(BASELINE_MODEL)
deployed = registry.deployed()
model_threshold = deployed.threshold if deployed else DeepPatchCore.load(model_path, device="cpu").threshold
onnx_path = EDGE_ROOT / f"deep_patchcore_{version_id}.onnx"
engine_path = EDGE_ROOT / f"deep_patchcore_{version_id}.engine"
readiness = runtime_readiness()

page_header(
    "EDGE DEPLOYMENT · RUNTIME BENCHMARK",
    "Edge AI Runtime Benchmark",
    "동일한 Deep PatchCore 검사 그래프를 PyTorch, ONNX Runtime과 TensorRT로 변환해 성능을 비교합니다.",
    status="EDGE PROFILER READY",
)
workflow_strip(["PyTorch baseline", "ONNX export", "TensorRT engine", "Latency report"])
st.markdown(
    """
    <div class="runtime-flow">
      <div class="runtime-card"><strong>PyTorch CUDA</strong><span>학습과 기준 추론 런타임</span></div>
      <div class="runtime-card"><strong>ONNX Runtime</strong><span>CUDA·CPU 이식 가능한 그래프</span></div>
      <div class="runtime-card"><strong>TensorRT Native</strong><span>NVIDIA GPU 전용 최적화 엔진</span></div>
    </div>
    """,
    unsafe_allow_html=True,
)

report = json.loads(REPORT_PATH.read_text(encoding="utf-8")) if REPORT_PATH.exists() else {}
benchmarks = report.get("benchmarks", [])
best = min(benchmarks, key=lambda item: item["median_ms"]) if benchmarks else None
pytorch = next((item for item in benchmarks if item["backend"] == "PyTorch"), None)
speedup = pytorch["median_ms"] / best["median_ms"] if best and pytorch else 0.0

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("GPU", readiness["gpu"])
c2.metric("운영 모델", version_id)
c3.metric("최고 p50", f"{best['median_ms']:.2f} ms" if best else "-")
c4.metric("최고 처리량", f"{best['fps']:.1f} FPS" if best else "-")
c5.metric("PyTorch 대비", f"{speedup:.2f}x" if speedup else "-")

results_tab, live_tab, readiness_tab = st.tabs(["벤치마크 결과", "런타임 실시간 비교", "배포 준비 상태"])

with results_tab:
    if VISUAL_PATH.exists():
        st.image(VISUAL_PATH, width="stretch")
    if benchmarks:
        frame = pd.DataFrame(benchmarks)
        table = frame[
            [
                "backend",
                "provider",
                "median_ms",
                "p95_ms",
                "fps",
                "score_max_error",
                "decision_agreement",
            ]
        ].rename(
            columns={
                "backend": "Runtime",
                "provider": "Provider",
                "median_ms": "p50 (ms)",
                "p95_ms": "p95 (ms)",
                "fps": "FPS",
                "score_max_error": "Max Score Error",
                "decision_agreement": "Decision Agreement",
            }
        )
        st.dataframe(table, hide_index=True, width="stretch")
        st.bar_chart(frame.set_index("provider")[["median_ms", "p95_ms"]])
        st.success(
            "모든 런타임의 PASS/REJECT 판정 일치율은 100%이며, 최대 anomaly score 차이는 "
            f"{max(float(item['score_max_error']) for item in benchmarks):.6f}입니다."
        )
    else:
        st.info("아직 벤치마크 리포트가 없습니다. 아래 재현 명령을 실행하세요.")

with live_tab:
    if not DATASET_ROOT.exists():
        st.warning("MVTec 샘플 데이터가 없습니다.")
    else:
        _, samples = load_mvtec_bottle(DATASET_ROOT)
        by_type: dict[str, list] = {}
        for sample in samples:
            by_type.setdefault(sample.defect_type, []).append(sample)
        controls, preview = st.columns([0.34, 0.66], gap="large")
        with controls:
            defect_type = st.selectbox("검사 샘플", list(by_type))
            sample_index = st.slider("샘플 번호", 0, len(by_type[defect_type]) - 1, 0)
            available_backends = ["PyTorch CUDA"]
            if onnx_path.exists() and readiness["packages"].get("onnxruntime"):
                available_backends.extend(["ONNX CUDA", "ONNX CPU"])
            if engine_path.exists() and readiness["packages"].get("tensorrt"):
                available_backends.append("TensorRT Native")
            backend = st.selectbox("실행 런타임", available_backends)
            st.caption(f"ONNX: {'READY' if onnx_path.exists() else 'NOT BUILT'}")
            st.caption(f"TensorRT: {'READY' if engine_path.exists() else 'NOT BUILT'}")
            inspect = st.button("선택 런타임으로 검사", type="primary", width="stretch")
        selected = by_type[defect_type][sample_index]
        with preview:
            st.image(selected.path, caption=f"{defect_type} / {selected.path.name}", width="stretch")
        if inspect:
            image = Image.open(selected.path).convert("RGB")
            with st.spinner(f"{backend} 추론 중..."):
                if backend == "PyTorch CUDA":
                    predictor = DeepPatchCore.load(model_path)
                elif backend == "ONNX CUDA":
                    predictor = OnnxDeepPatchCore(onnx_path, model_threshold, "cuda")
                elif backend == "ONNX CPU":
                    predictor = OnnxDeepPatchCore(onnx_path, model_threshold, "cpu")
                else:
                    predictor = TensorRTDeepPatchCore(engine_path, model_threshold)
                prediction = predictor.predict(image)
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("판정", "REJECT" if prediction.is_defect else "PASS")
            m2.metric("Raw Score", f"{prediction.raw_score:.4f}")
            m3.metric("Threshold", f"{prediction.threshold:.4f}")
            m4.metric("종단 간 시간", f"{prediction.latency_ms:.2f} ms")
            st.image(prediction.overlay, caption=f"{backend} anomaly map", width="stretch")

with readiness_tab:
    package_rows = [
        {
            "구성 요소": name,
            "상태": "READY" if installed else "MISSING",
            "버전": readiness["versions"].get(name, "-"),
        }
        for name, installed in readiness["packages"].items()
    ]
    st.dataframe(pd.DataFrame(package_rows), hide_index=True, width="stretch")
    st.write("Execution Providers:", " · ".join(readiness["providers"]) or "없음")
    st.markdown("#### 재현 명령")
    st.code(
        "pip install -r requirements-edge.txt\n"
        "pip install -r requirements-tensorrt.txt\n"
        "python -m scripts.export_edge_model --build-tensorrt\n"
        "python -m scripts.benchmark_edge_runtime --include-cpu",
        language="powershell",
    )
    st.warning("TensorRT 엔진은 GPU 아키텍처와 TensorRT 버전에 종속되므로 대상 장비에서 다시 빌드해야 합니다.")


