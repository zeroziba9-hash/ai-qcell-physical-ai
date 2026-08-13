from __future__ import annotations

import argparse
import json
from pathlib import Path

from qcell.active_learning import (
    TrainingConfig,
    ensure_baseline_registered,
    train_and_register,
)
from qcell.dataset_studio import DatasetStudio
from qcell.model_registry import ModelRegistry


ROOT = Path(__file__).resolve().parents[1]
ACTIVE_ROOT = ROOT / "artifacts" / "active_learning"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an AI-QCell active-learning model")
    parser.add_argument("--seed-demo", action="store_true", help="import 52 MVTec demo images")
    parser.add_argument("--deploy", action="store_true", help="deploy the trained model")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--candidate-size", type=int, default=4000)
    parser.add_argument("--coreset-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--name", default="AI-QCell Active Learning Demo")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = DatasetStudio(ACTIVE_ROOT / "dataset")
    registry = ModelRegistry(ACTIVE_ROOT / "model_registry")
    ensure_baseline_registered(
        registry,
        ROOT / "models" / "deep_patchcore_bottle.pt",
        ROOT / "models" / "deep_patchcore_bottle.json",
        ROOT / "docs" / "results" / "deep_patchcore_bottle_report.json",
    )
    if args.seed_demo:
        added = dataset.seed_from_mvtec(
            ROOT / "data" / "mvtec-ad" / "bottle",
            normal_count=40,
            defect_count=12,
        )
        print(f"dataset_added={added}")
    result = train_and_register(
        dataset,
        registry,
        TrainingConfig(
            batch_size=args.batch_size,
            candidate_size=args.candidate_size,
            coreset_size=args.coreset_size,
            seed=args.seed,
        ),
        display_name=args.name,
    )
    if args.deploy:
        registry.deploy(result.version.version_id, reason="cli-auto-deploy")
    print(
        json.dumps(
            {
                "version_id": result.version.version_id,
                "threshold": result.version.threshold,
                "metrics": result.metrics,
                "deployed": registry.deployed_version_id(),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
