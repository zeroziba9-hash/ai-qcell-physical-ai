from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from qcell.edge_runtime import build_tensorrt_engine, export_deep_patchcore_onnx
from qcell.model_registry import ModelRegistry


ROOT = Path(__file__).resolve().parents[1]
EDGE_ROOT = ROOT / "artifacts" / "edge_runtime"
REGISTRY_ROOT = ROOT / "artifacts" / "active_learning" / "model_registry"
BASELINE_MODEL = ROOT / "models" / "deep_patchcore_bottle.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export AI-QCell Deep PatchCore to ONNX")
    parser.add_argument("--build-tensorrt", action="store_true")
    parser.add_argument("--workspace-gib", type=float, default=2.0)
    return parser.parse_args()


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = parse_args()
    registry = ModelRegistry(REGISTRY_ROOT)
    model_path, version_id = registry.resolve_model_path(BASELINE_MODEL)
    onnx_path = EDGE_ROOT / f"deep_patchcore_{version_id}.onnx"
    metadata = export_deep_patchcore_onnx(
        model_path,
        onnx_path,
        model_version=version_id,
    )
    result: dict[str, object] = {"onnx": metadata.to_dict()}
    if args.build_tensorrt:
        engine_path = EDGE_ROOT / f"deep_patchcore_{version_id}.engine"
        build_tensorrt_engine(onnx_path, engine_path, workspace_gib=args.workspace_gib)
        result["tensorrt"] = {
            "engine_path": str(engine_path.resolve()),
            "file_size_bytes": engine_path.stat().st_size,
        }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
