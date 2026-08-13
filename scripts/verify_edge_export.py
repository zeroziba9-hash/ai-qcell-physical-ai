from __future__ import annotations

import argparse
import json
from pathlib import Path

from qcell.deep_patchcore import DeepPatchCore
from qcell.edge_runtime import OnnxDeepPatchCore
from qcell.model_registry import ModelRegistry
from qcell.vision import generate_demo_pair


ROOT = Path(__file__).resolve().parents[1]
EDGE_ROOT = ROOT / "artifacts" / "edge_runtime"
REGISTRY_ROOT = ROOT / "artifacts" / "active_learning" / "model_registry"
BASELINE_MODEL = ROOT / "models" / "deep_patchcore_bottle.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify PyTorch and ONNX score equivalence")
    parser.add_argument("--provider", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--max-score-error", type=float, default=0.001)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    registry = ModelRegistry(REGISTRY_ROOT)
    model_path, version_id = registry.resolve_model_path(BASELINE_MODEL)
    onnx_path = EDGE_ROOT / f"deep_patchcore_{version_id}.onnx"
    if not onnx_path.exists():
        raise FileNotFoundError(onnx_path)
    pytorch = DeepPatchCore.load(model_path, device="cpu" if args.provider == "cpu" else None)
    onnx_runtime = OnnxDeepPatchCore(onnx_path, pytorch.threshold, provider=args.provider)
    normal, scratch = generate_demo_pair("scratch")
    samples = [normal, scratch]
    comparisons = []
    for index, image in enumerate(samples):
        reference = pytorch.predict(image)
        candidate = onnx_runtime.predict(image)
        comparisons.append(
            {
                "sample": index,
                "pytorch_score": reference.raw_score,
                "onnx_score": candidate.raw_score,
                "absolute_error": round(abs(reference.raw_score - candidate.raw_score), 8),
                "decision_match": reference.is_defect == candidate.is_defect,
            }
        )
    max_error = max(item["absolute_error"] for item in comparisons)
    decision_match = all(item["decision_match"] for item in comparisons)
    payload = {
        "model_version": version_id,
        "provider": onnx_runtime.provider,
        "max_score_error": max_error,
        "decision_match": decision_match,
        "comparisons": comparisons,
    }
    print(json.dumps(payload, indent=2))
    if max_error > args.max_score_error or not decision_match:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
