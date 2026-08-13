from __future__ import annotations

import json
from pathlib import Path
import random

from qcell.deep_patchcore import DeepPatchCore, load_mvtec_bottle


def main() -> None:
    train_paths, test_samples = load_mvtec_bottle("data/mvtec-ad/bottle")
    if not train_paths:
        raise FileNotFoundError(
            "MVTec bottle data not found. Run python -m scripts.download_mvtec_bottle"
        )

    shuffled = train_paths.copy()
    random.Random(42).shuffle(shuffled)
    validation_paths = shuffled[:32]
    memory_paths = shuffled[32:]

    model = DeepPatchCore()
    metadata = model.fit(
        memory_paths,
        validation_paths,
        batch_size=16,
        candidate_size=14000,
        coreset_size=900,
        seed=42,
    )
    metadata.update(
        {
            "dataset": "MVTec AD",
            "category": "bottle",
            "test_images": len(test_samples),
            "dataset_license": "CC BY-NC-SA 4.0",
            "method": "PatchCore-style ImageNet feature memory bank",
            "official_reference": "https://anomalib.readthedocs.io/en/latest/markdown/guides/reference/models/image/patchcore.html",
        }
    )
    model_path = model.save("models/deep_patchcore_bottle.pt", metadata)
    metadata_path = Path("models/deep_patchcore_bottle.json")
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))
    print(f"model={model_path.resolve()}")


if __name__ == "__main__":
    main()
