from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import shutil
from threading import RLock


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def new_version_id(prefix: str = "custom") -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return f"{prefix}-{stamp}"


@dataclass(frozen=True)
class ModelVersion:
    version_id: str
    display_name: str
    model_path: str
    metadata_path: str
    dataset_fingerprint: str
    threshold: float
    metrics: dict[str, object]
    training_seconds: float
    created_at: str

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "ModelVersion":
        return cls(
            version_id=str(payload.get("version_id", "")),
            display_name=str(payload.get("display_name", "")),
            model_path=str(payload.get("model_path", "")),
            metadata_path=str(payload.get("metadata_path", "")),
            dataset_fingerprint=str(payload.get("dataset_fingerprint", "")),
            threshold=float(payload.get("threshold", 0.0)),
            metrics=dict(payload.get("metrics", {})),
            training_seconds=float(payload.get("training_seconds", 0.0)),
            created_at=str(payload.get("created_at", "")),
        )

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class ModelRegistry:
    schema_version = 1

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.versions_dir = self.root / "versions"
        self.manifest_path = self.root / "registry.json"
        self._lock = RLock()

    def _read(self) -> dict[str, object]:
        if not self.manifest_path.exists():
            return {
                "schema_version": self.schema_version,
                "updated_at": _utc_now(),
                "deployed_version": "",
                "deployment_history": [],
                "versions": [],
            }
        payload = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        if int(payload.get("schema_version", 0)) != self.schema_version:
            raise ValueError("unsupported model registry schema")
        return payload

    def _write(self, payload: dict[str, object]) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        payload["schema_version"] = self.schema_version
        payload["updated_at"] = _utc_now()
        temporary = self.manifest_path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        temporary.replace(self.manifest_path)

    def versions(self) -> list[ModelVersion]:
        versions = [ModelVersion.from_dict(item) for item in self._read().get("versions", [])]
        return sorted(versions, key=lambda version: version.created_at, reverse=True)

    def get(self, version_id: str) -> ModelVersion:
        for version in self.versions():
            if version.version_id == version_id:
                return version
        raise KeyError(f"model version not found: {version_id}")

    def register(
        self,
        source_model: str | Path,
        *,
        version_id: str,
        display_name: str,
        metadata: dict[str, object],
        metrics: dict[str, object],
        dataset_fingerprint: str,
        threshold: float,
        training_seconds: float,
    ) -> ModelVersion:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{2,63}", version_id):
            raise ValueError("version_id must be a safe 3-64 character slug")
        source = Path(source_model)
        if not source.is_file():
            raise FileNotFoundError(source)
        with self._lock:
            payload = self._read()
            if any(str(item.get("version_id")) == version_id for item in payload["versions"]):
                raise ValueError(f"model version already exists: {version_id}")
            version_dir = self.versions_dir / version_id
            version_dir.mkdir(parents=True, exist_ok=False)
            destination = version_dir / "model.pt"
            shutil.copy2(source, destination)
            metadata_path = version_dir / "metadata.json"
            metadata_payload = dict(metadata)
            metadata_payload.update(
                {
                    "version_id": version_id,
                    "display_name": display_name,
                    "dataset_fingerprint": dataset_fingerprint,
                    "metrics": metrics,
                    "threshold": float(threshold),
                    "training_seconds": float(training_seconds),
                    "registered_at": _utc_now(),
                }
            )
            metadata_path.write_text(
                json.dumps(metadata_payload, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            version = ModelVersion(
                version_id=version_id,
                display_name=display_name,
                model_path=destination.relative_to(self.root).as_posix(),
                metadata_path=metadata_path.relative_to(self.root).as_posix(),
                dataset_fingerprint=dataset_fingerprint,
                threshold=float(threshold),
                metrics=metrics,
                training_seconds=float(training_seconds),
                created_at=_utc_now(),
            )
            payload["versions"].append(version.to_dict())
            self._write(payload)
        return version

    def deployed_version_id(self) -> str:
        return str(self._read().get("deployed_version", ""))

    def deployed(self) -> ModelVersion | None:
        version_id = self.deployed_version_id()
        return self.get(version_id) if version_id else None

    def deploy(self, version_id: str, reason: str = "manual") -> ModelVersion:
        version = self.get(version_id)
        if not (self.root / version.model_path).is_file():
            raise FileNotFoundError(self.root / version.model_path)
        with self._lock:
            payload = self._read()
            payload["deployed_version"] = version_id
            history = list(payload.get("deployment_history", []))
            history.append(
                {"version_id": version_id, "deployed_at": _utc_now(), "reason": reason}
            )
            payload["deployment_history"] = history[-50:]
            self._write(payload)
        return version

    def rollback(self) -> ModelVersion:
        payload = self._read()
        current = str(payload.get("deployed_version", ""))
        history = list(payload.get("deployment_history", []))
        previous = next(
            (
                str(item.get("version_id", ""))
                for item in reversed(history[:-1] if history else [])
                if str(item.get("version_id", "")) and str(item.get("version_id")) != current
            ),
            "",
        )
        if not previous:
            raise ValueError("no previous deployment is available")
        return self.deploy(previous, reason=f"rollback-from:{current}")

    def deployment_history(self) -> list[dict[str, str]]:
        return [
            {str(key): str(value) for key, value in item.items()}
            for item in reversed(self._read().get("deployment_history", []))
        ]

    def resolve_model_path(self, fallback: str | Path) -> tuple[Path, str]:
        deployed = self.deployed()
        if deployed is not None:
            path = self.root / deployed.model_path
            if path.is_file():
                return path, deployed.version_id
        return Path(fallback), "baseline"
