from pathlib import Path

from qcell.model_registry import ModelRegistry


def register(registry: ModelRegistry, source: Path, version_id: str, f1: float):
    return registry.register(
        source,
        version_id=version_id,
        display_name=version_id,
        metadata={"backbone": "fake"},
        metrics={"f1": f1},
        dataset_fingerprint=f"data-{version_id}",
        threshold=0.5,
        training_seconds=1.0,
    )


def test_registry_deploy_and_rollback(tmp_path: Path) -> None:
    source = tmp_path / "model.pt"
    source.write_bytes(b"model")
    registry = ModelRegistry(tmp_path / "registry")
    first = register(registry, source, "model-v1", 0.8)
    second = register(registry, source, "model-v2", 0.9)

    registry.deploy(first.version_id)
    registry.deploy(second.version_id)
    assert registry.deployed_version_id() == "model-v2"
    assert registry.resolve_model_path(source)[1] == "model-v2"

    rolled_back = registry.rollback()
    assert rolled_back.version_id == "model-v1"
    assert registry.deployed_version_id() == "model-v1"
    assert len(registry.deployment_history()) == 3


def test_registry_falls_back_without_deployment(tmp_path: Path) -> None:
    fallback = tmp_path / "fallback.pt"
    fallback.write_bytes(b"baseline")
    registry = ModelRegistry(tmp_path / "registry")
    path, version = registry.resolve_model_path(fallback)
    assert path == fallback
    assert version == "baseline"
