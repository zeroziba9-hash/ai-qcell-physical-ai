from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter

from qcell.patch_memory import PatchMemoryDetector, generate_normal_training_set


def main() -> None:
    started = perf_counter()
    training_images = generate_normal_training_set(count=40, seed=42)
    model = PatchMemoryDetector.fit(training_images, image_size=(256, 256), grid_size=16)
    model_path = model.save("models/patch_memory_demo.npz")

    metadata = {
        "model": "PatchMemoryDetector",
        "training_type": "unsupervised_normal_only",
        "training_images": len(training_images),
        "image_size": list(model.image_size),
        "grid_size": model.grid_size,
        "patches_per_image": model.grid_size**2,
        "feature_dimensions": int(model.memory.shape[-1]),
        "patch_threshold": round(model.patch_threshold, 6),
        "image_threshold": round(model.image_threshold, 6),
        "training_seconds": round(perf_counter() - started, 3),
        "note": "Synthetic aligned-product baseline; replace with MVTec AD + deep PatchCore.",
    }
    metadata_path = Path("models/patch_memory_demo.json")
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"model={model_path.resolve()}")
    print(f"metadata={metadata_path.resolve()}")


if __name__ == "__main__":
    main()
