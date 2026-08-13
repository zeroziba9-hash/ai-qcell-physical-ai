from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from qcell.deep_patchcore import DeepPatchCore, load_mvtec_bottle
from qcell.edge_runtime import (
    OnnxDeepPatchCore,
    TensorRTDeepPatchCore,
    benchmark_predictor,
    runtime_readiness,
    write_benchmark_report,
)
from qcell.model_registry import ModelRegistry


ROOT = Path(__file__).resolve().parents[1]
EDGE_ROOT = ROOT / "artifacts" / "edge_runtime"
REGISTRY_ROOT = ROOT / "artifacts" / "active_learning" / "model_registry"
BASELINE_MODEL = ROOT / "models" / "deep_patchcore_bottle.pt"
DATASET_ROOT = ROOT / "data" / "mvtec-ad" / "bottle"
REPORT_PATH = ROOT / "docs" / "results" / "edge_runtime_benchmark.json"
VISUAL_PATH = ROOT / "docs" / "images" / "edge_runtime_benchmark.png"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark AI-QCell edge runtimes")
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--include-cpu", action="store_true")
    return parser.parse_args()


def _samples() -> list[Image.Image]:
    _, samples = load_mvtec_bottle(DATASET_ROOT)
    selected = []
    for defect_type in ["good", "broken_large", "broken_small", "contamination"]:
        sample = next(item for item in samples if item.defect_type == defect_type)
        selected.append(Image.open(sample.path).convert("RGB"))
    return selected


def main() -> None:
    args = parse_args()
    registry = ModelRegistry(REGISTRY_ROOT)
    model_path, version_id = registry.resolve_model_path(BASELINE_MODEL)
    onnx_path = EDGE_ROOT / f"deep_patchcore_{version_id}.onnx"
    engine_path = EDGE_ROOT / f"deep_patchcore_{version_id}.engine"
    if not onnx_path.exists():
        raise FileNotFoundError(
            f"ONNX model not found: {onnx_path}. Run python -m scripts.export_edge_model first."
        )

    images = _samples()
    pytorch = DeepPatchCore.load(model_path)
    references = [pytorch.predict(image) for image in images]
    reference_scores = [prediction.raw_score for prediction in references]
    reference_decisions = [prediction.is_defect for prediction in references]
    common = {
        "images": images,
        "reference_scores": reference_scores,
        "reference_decisions": reference_decisions,
        "warmup_runs": args.warmup,
        "measured_runs": args.runs,
    }
    benchmarks = [
        benchmark_predictor(
            "PyTorch",
            str(pytorch.device),
            pytorch,
            **common,
        )
    ]
    cuda = OnnxDeepPatchCore(onnx_path, pytorch.threshold, provider="cuda")
    benchmarks.append(
        benchmark_predictor("ONNX Runtime", cuda.provider, cuda, **common)
    )
    if args.include_cpu:
        cpu = OnnxDeepPatchCore(onnx_path, pytorch.threshold, provider="cpu")
        benchmarks.append(
            benchmark_predictor("ONNX Runtime", cpu.provider, cpu, **common)
        )
    if engine_path.exists():
        tensorrt = TensorRTDeepPatchCore(engine_path, pytorch.threshold)
        benchmarks.append(
            benchmark_predictor("TensorRT", tensorrt.provider, tensorrt, **common)
        )

    write_benchmark_report(
        REPORT_PATH,
        benchmarks,
        {
            "model_version": version_id,
            "model_path": _display_path(model_path),
            "onnx_path": _display_path(onnx_path),
            "engine_path": _display_path(engine_path) if engine_path.exists() else "",
            "sample_types": ["good", "broken_large", "broken_small", "contamination"],
            "measurement_scope": "end-to-end image preparation, inference and overlay",
        },
    )
    render_report(benchmarks, version_id, VISUAL_PATH)
    print(REPORT_PATH.read_text(encoding="utf-8"))
    print(f"visual={VISUAL_PATH.resolve()}")


def _display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path.resolve())


def render_report(benchmarks, version_id: str, path: Path) -> None:
    canvas = Image.new("RGB", (1400, 820), "#050b16")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.truetype("arial.ttf", 25)
    bold = ImageFont.truetype("arialbd.ttf", 44)
    small = ImageFont.truetype("arial.ttf", 20)
    draw.text((70, 55), "AI-QCell Edge Runtime Benchmark", font=bold, fill="#e2e8f0")
    draw.text((72, 115), f"Model {version_id} · RTX 4080 SUPER · End-to-end", font=font, fill="#67e8f9")
    colors = ["#60a5fa", "#22d3ee", "#a78bfa", "#4ade80"]
    max_latency = max(benchmark.median_ms for benchmark in benchmarks)
    for index, benchmark in enumerate(benchmarks):
        y = 205 + index * 135
        label = f"{benchmark.backend} · {benchmark.provider}"
        draw.text((75, y), label, font=font, fill="#f8fafc")
        width = int(760 * benchmark.median_ms / max(max_latency, 1e-6))
        draw.rounded_rectangle((75, y + 42, 75 + width, y + 88), 12, fill=colors[index % len(colors)])
        draw.text((860, y + 42), f"p50 {benchmark.median_ms:.2f} ms", font=font, fill="#f8fafc")
        draw.text((1110, y + 42), f"{benchmark.fps:.1f} FPS", font=font, fill="#86efac")
        draw.text(
            (860, y + 82),
            f"score error {benchmark.score_max_error:.6f} · decision {benchmark.decision_agreement * 100:.0f}%",
            font=small,
            fill="#94a3b8",
        )
    readiness = runtime_readiness()
    draw.text(
        (75, 750),
        f"CUDA {readiness['cuda_version']} · TensorRT {readiness['versions'].get('tensorrt', 'N/A')} · ONNX Runtime {readiness['versions'].get('onnxruntime', 'N/A')}",
        font=small,
        fill="#94a3b8",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path, quality=94)


if __name__ == "__main__":
    main()
