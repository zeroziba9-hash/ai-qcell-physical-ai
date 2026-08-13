from __future__ import annotations

from pathlib import Path

from huggingface_hub import snapshot_download


def main() -> None:
    output = Path("data/mvtec-ad")
    output.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id="foersben/mvtec-ad",
        repo_type="dataset",
        allow_patterns=["bottle/**"],
        local_dir=output,
    )
    bottle = output / "bottle"
    train_count = len(list((bottle / "train" / "good").glob("*.png")))
    test_count = len(list((bottle / "test").glob("*/*.png")))
    print(f"dataset={bottle.resolve()}")
    print(f"train_good={train_count}")
    print(f"test_total={test_count}")


if __name__ == "__main__":
    main()
