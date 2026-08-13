from __future__ import annotations

from dataclasses import asdict, dataclass
import importlib.util
import json
import math
from pathlib import Path
import platform
from statistics import mean, median
from time import perf_counter
from typing import Callable, Iterable, Protocol

import numpy as np
from PIL import Image
import torch
from torch import Tensor, nn
import torch.nn.functional as functional

from .deep_patchcore import DeepPatchCore, render_deep_overlay


@dataclass(frozen=True)
class EdgePrediction:
    is_defect: bool
    anomaly_score: float
    raw_score: float
    threshold: float
    latency_ms: float
    anomaly_map: np.ndarray
    overlay: Image.Image
    prepared_image: Image.Image
    backend: str


@dataclass(frozen=True)
class RuntimeBenchmark:
    backend: str
    provider: str
    sample_count: int
    warmup_runs: int
    measured_runs: int
    mean_ms: float
    median_ms: float
    p95_ms: float
    min_ms: float
    max_ms: float
    fps: float
    score_mae: float
    score_max_error: float
    decision_agreement: float
    scores: tuple[float, ...]

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["scores"] = list(self.scores)
        return payload


@dataclass(frozen=True)
class EdgeArtifactMetadata:
    onnx_path: str
    model_path: str
    model_version: str
    threshold: float
    memory_bank_patches: int
    feature_dimensions: int
    opset: int
    input_shape: tuple[int, int, int, int]
    output_names: tuple[str, str]
    file_size_bytes: int

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["input_shape"] = list(self.input_shape)
        payload["output_names"] = list(self.output_names)
        return payload


class Predictor(Protocol):
    def predict(self, image: Image.Image) -> object: ...


class DeepPatchCoreExportModule(nn.Module):
    """Full PatchCore inference graph: preprocessing, backbone and nearest patches."""

    def __init__(self, model: DeepPatchCore) -> None:
        super().__init__()
        if model.memory_bank is None:
            raise ValueError("Deep PatchCore memory bank is required for export")
        self.extractor = model.extractor.to("cpu").eval()
        self.register_buffer("memory_bank", model.memory_bank.detach().cpu().float())
        self.register_buffer(
            "normalization_mean", torch.tensor([0.485, 0.456, 0.406])[None, :, None, None]
        )
        self.register_buffer(
            "normalization_std", torch.tensor([0.229, 0.224, 0.225])[None, :, None, None]
        )

    def forward(self, images: Tensor) -> tuple[Tensor, Tensor]:
        normalized = (images - self.normalization_mean) / self.normalization_std
        features = self.extractor(normalized)
        layer2 = functional.avg_pool2d(features["layer2"], kernel_size=3, stride=1, padding=1)
        layer3 = functional.avg_pool2d(features["layer3"], kernel_size=3, stride=1, padding=1)
        layer3 = functional.interpolate(
            layer3, size=(28, 28), mode="bilinear", align_corners=False
        )
        combined = torch.cat([layer2, layer3], dim=1)
        combined = functional.normalize(combined, p=2, dim=1)
        embeddings = combined.permute(0, 2, 3, 1).reshape(images.shape[0], 784, 384)
        patch_scores = nearest_patch_distances(embeddings, self.memory_bank)
        image_scores = patch_scores.amax(dim=1)
        return image_scores, patch_scores.reshape(images.shape[0], 28, 28)


def nearest_patch_distances(embeddings: Tensor, memory_bank: Tensor) -> Tensor:
    """ONNX/TensorRT-friendly Euclidean nearest-neighbor distance."""
    embedding_norm = embeddings.square().sum(dim=-1, keepdim=True)
    memory_norm = memory_bank.square().sum(dim=-1).reshape(1, 1, -1)
    squared = embedding_norm + memory_norm - 2.0 * torch.matmul(
        embeddings, memory_bank.transpose(0, 1)
    )
    return torch.sqrt(torch.clamp(squared, min=0.0)).amin(dim=-1)


def image_to_input(image: Image.Image) -> np.ndarray:
    prepared = DeepPatchCore.prepare_image(image)
    array = np.asarray(prepared, dtype=np.float32) / 255.0
    return np.ascontiguousarray(array.transpose(2, 0, 1)[None])


def export_deep_patchcore_onnx(
    model_path: str | Path,
    output_path: str | Path,
    *,
    model_version: str = "baseline",
    opset: int = 18,
    metadata_path: str | Path | None = None,
) -> EdgeArtifactMetadata:
    source = Path(model_path)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    model = DeepPatchCore.load(source, device="cpu")
    module = DeepPatchCoreExportModule(model).eval()
    example = torch.zeros(1, 3, 224, 224, dtype=torch.float32)
    torch.onnx.export(
        module,
        (example,),
        str(output),
        input_names=["images"],
        output_names=["image_scores", "anomaly_maps"],
        opset_version=opset,
        dynamo=True,
        external_data=False,
    )
    import onnx

    graph = onnx.load(str(output))
    onnx.checker.check_model(graph)
    metadata = EdgeArtifactMetadata(
        onnx_path=str(output.resolve()),
        model_path=str(source.resolve()),
        model_version=model_version,
        threshold=model.threshold,
        memory_bank_patches=len(model.memory_bank),
        feature_dimensions=DeepPatchCore.feature_dimensions,
        opset=opset,
        input_shape=(1, 3, 224, 224),
        output_names=("image_scores", "anomaly_maps"),
        file_size_bytes=output.stat().st_size,
    )
    destination = Path(metadata_path) if metadata_path else output.with_suffix(".json")
    destination.write_text(json.dumps(metadata.to_dict(), indent=2), encoding="utf-8")
    return metadata


class OnnxDeepPatchCore:
    def __init__(
        self,
        onnx_path: str | Path,
        threshold: float,
        provider: str = "cuda",
        cache_dir: str | Path | None = None,
    ) -> None:
        import onnxruntime as ort

        if hasattr(ort, "preload_dlls"):
            ort.preload_dlls()
        available = ort.get_available_providers()
        provider_key = provider.lower()
        if provider_key == "cuda":
            if "CUDAExecutionProvider" not in available:
                raise RuntimeError("ONNX Runtime CUDAExecutionProvider is unavailable")
            providers: list[object] = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        elif provider_key == "tensorrt":
            if "TensorrtExecutionProvider" not in available:
                raise RuntimeError("ONNX Runtime TensorrtExecutionProvider is unavailable")
            cache = Path(cache_dir or Path(onnx_path).parent / "trt_cache")
            cache.mkdir(parents=True, exist_ok=True)
            providers = [
                (
                    "TensorrtExecutionProvider",
                    {
                        "trt_engine_cache_enable": "1",
                        "trt_engine_cache_path": str(cache.resolve()),
                    },
                ),
                "CUDAExecutionProvider",
                "CPUExecutionProvider",
            ]
        elif provider_key == "cpu":
            providers = ["CPUExecutionProvider"]
        else:
            raise ValueError("provider must be cpu, cuda or tensorrt")
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(
            str(onnx_path), sess_options=options, providers=providers
        )
        self.threshold = float(threshold)
        self.provider = self.session.get_providers()[0]
        self.backend = f"ONNX Runtime · {self.provider}"

    def predict(self, image: Image.Image) -> EdgePrediction:
        started = perf_counter()
        prepared = DeepPatchCore.prepare_image(image)
        image_scores, anomaly_maps = self.session.run(
            ["image_scores", "anomaly_maps"], {"images": image_to_input(prepared)}
        )
        return make_edge_prediction(
            prepared,
            float(image_scores[0]),
            np.asarray(anomaly_maps[0], dtype=np.float32),
            self.threshold,
            (perf_counter() - started) * 1000.0,
            self.backend,
        )


def build_tensorrt_engine(
    onnx_path: str | Path,
    engine_path: str | Path,
    workspace_gib: float = 2.0,
) -> Path:
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
    )
    parser = trt.OnnxParser(network, logger)
    if not parser.parse(Path(onnx_path).read_bytes()):
        errors = [str(parser.get_error(index)) for index in range(parser.num_errors)]
        raise RuntimeError("TensorRT ONNX parse failed:\n" + "\n".join(errors))
    config = builder.create_builder_config()
    config.set_memory_pool_limit(
        trt.MemoryPoolType.WORKSPACE, int(workspace_gib * 1024**3)
    )
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("TensorRT engine build failed")
    output = Path(engine_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(bytes(serialized))
    return output


class TensorRTDeepPatchCore:
    def __init__(self, engine_path: str | Path, threshold: float) -> None:
        import tensorrt as trt

        if not torch.cuda.is_available():
            raise RuntimeError("TensorRT requires a CUDA device")
        self._trt = trt
        self._logger = trt.Logger(trt.Logger.WARNING)
        self._runtime = trt.Runtime(self._logger)
        self.engine = self._runtime.deserialize_cuda_engine(Path(engine_path).read_bytes())
        if self.engine is None:
            raise RuntimeError("could not deserialize TensorRT engine")
        self.context = self.engine.create_execution_context()
        self.threshold = float(threshold)
        self.provider = "TensorRT 11 Native"
        self.backend = self.provider
        self.stream = torch.cuda.Stream()
        self.input_name = next(
            self.engine.get_tensor_name(index)
            for index in range(self.engine.num_io_tensors)
            if self.engine.get_tensor_mode(self.engine.get_tensor_name(index))
            == trt.TensorIOMode.INPUT
        )
        self.output_names = [
            self.engine.get_tensor_name(index)
            for index in range(self.engine.num_io_tensors)
            if self.engine.get_tensor_mode(self.engine.get_tensor_name(index))
            == trt.TensorIOMode.OUTPUT
        ]

    def _run(self, array: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        input_tensor = torch.from_numpy(array).to("cuda").contiguous()
        self.context.set_input_shape(self.input_name, tuple(input_tensor.shape))
        tensors: dict[str, Tensor] = {self.input_name: input_tensor}
        for name in self.output_names:
            shape = tuple(int(value) for value in self.context.get_tensor_shape(name))
            dtype = _torch_dtype_from_trt(self.engine.get_tensor_dtype(name), self._trt)
            tensors[name] = torch.empty(shape, dtype=dtype, device="cuda")
        for name, tensor in tensors.items():
            if not self.context.set_tensor_address(name, tensor.data_ptr()):
                raise RuntimeError(f"could not bind TensorRT tensor: {name}")
        self.stream.wait_stream(torch.cuda.current_stream())
        stream_handle = self.stream.cuda_stream
        if not self.context.execute_async_v3(stream_handle=stream_handle):
            raise RuntimeError("TensorRT execution failed")
        self.stream.synchronize()
        outputs = [tensors[name].detach().cpu().numpy() for name in self.output_names]
        by_name = dict(zip(self.output_names, outputs))
        return by_name["image_scores"], by_name["anomaly_maps"]

    def predict(self, image: Image.Image) -> EdgePrediction:
        started = perf_counter()
        prepared = DeepPatchCore.prepare_image(image)
        image_scores, anomaly_maps = self._run(image_to_input(prepared))
        return make_edge_prediction(
            prepared,
            float(image_scores[0]),
            np.asarray(anomaly_maps[0], dtype=np.float32),
            self.threshold,
            (perf_counter() - started) * 1000.0,
            self.backend,
        )


def _torch_dtype_from_trt(dtype: object, trt: object) -> torch.dtype:
    mapping = {
        trt.float32: torch.float32,
        trt.float16: torch.float16,
        trt.int32: torch.int32,
        trt.int8: torch.int8,
        trt.bool: torch.bool,
    }
    if dtype not in mapping:
        raise TypeError(f"unsupported TensorRT output dtype: {dtype}")
    return mapping[dtype]


def make_edge_prediction(
    prepared: Image.Image,
    raw_score: float,
    coarse_map: np.ndarray,
    threshold: float,
    latency_ms: float,
    backend: str,
) -> EdgePrediction:
    full_map = np.asarray(
        Image.fromarray(coarse_map.astype(np.float32)).resize(
            DeepPatchCore.image_size, Image.Resampling.BILINEAR
        ),
        dtype=np.float32,
    )
    is_defect = raw_score > threshold
    overlay = render_deep_overlay(prepared, full_map, threshold, is_defect)
    anomaly_score = min(100.0, raw_score / max(threshold, 1e-8) * 50.0)
    return EdgePrediction(
        is_defect=is_defect,
        anomaly_score=round(anomaly_score, 1),
        raw_score=round(raw_score, 6),
        threshold=round(threshold, 6),
        latency_ms=round(latency_ms, 2),
        anomaly_map=full_map,
        overlay=overlay,
        prepared_image=prepared,
        backend=backend,
    )


def benchmark_predictor(
    backend: str,
    provider: str,
    predictor: Predictor,
    images: list[Image.Image],
    *,
    reference_scores: Iterable[float] | None = None,
    reference_decisions: Iterable[bool] | None = None,
    warmup_runs: int = 3,
    measured_runs: int = 10,
) -> RuntimeBenchmark:
    if not images:
        raise ValueError("at least one benchmark image is required")
    if warmup_runs < 0 or measured_runs < 1:
        raise ValueError("warmup_runs must be non-negative and measured_runs positive")
    for index in range(warmup_runs):
        predictor.predict(images[index % len(images)])

    durations: list[float] = []
    measured_predictions: list[object] = []
    for index in range(measured_runs):
        image = images[index % len(images)]
        started = perf_counter()
        prediction = predictor.predict(image)
        durations.append((perf_counter() - started) * 1000.0)
        measured_predictions.append(prediction)

    canonical_predictions = [predictor.predict(image) for image in images]
    scores = [float(prediction.raw_score) for prediction in canonical_predictions]
    decisions = [bool(prediction.is_defect) for prediction in canonical_predictions]
    reference_score_list = list(reference_scores or scores)
    reference_decision_list = list(reference_decisions or decisions)
    if len(reference_score_list) != len(scores) or len(reference_decision_list) != len(decisions):
        raise ValueError("reference results must match benchmark image count")
    errors = [abs(score - reference) for score, reference in zip(scores, reference_score_list)]
    agreement = mean(
        float(decision == reference)
        for decision, reference in zip(decisions, reference_decision_list)
    )
    p95 = float(np.percentile(np.asarray(durations), 95))
    median_ms = float(median(durations))
    return RuntimeBenchmark(
        backend=backend,
        provider=provider,
        sample_count=len(images),
        warmup_runs=warmup_runs,
        measured_runs=measured_runs,
        mean_ms=round(float(mean(durations)), 3),
        median_ms=round(median_ms, 3),
        p95_ms=round(p95, 3),
        min_ms=round(min(durations), 3),
        max_ms=round(max(durations), 3),
        fps=round(1000.0 / median_ms if median_ms > 0 else math.inf, 2),
        score_mae=round(float(mean(errors)), 8),
        score_max_error=round(max(errors), 8),
        decision_agreement=round(agreement, 4),
        scores=tuple(scores),
    )


def runtime_readiness() -> dict[str, object]:
    packages = {
        name: importlib.util.find_spec(name) is not None
        for name in ["onnx", "onnxscript", "onnxruntime", "tensorrt"]
    }
    providers: list[str] = []
    versions: dict[str, str] = {"torch": torch.__version__}
    if packages["onnxruntime"]:
        import onnxruntime as ort

        providers = ort.get_available_providers()
        versions["onnxruntime"] = ort.__version__
    if packages["onnx"]:
        import onnx

        versions["onnx"] = onnx.__version__
    if packages["tensorrt"]:
        import tensorrt as trt

        versions["tensorrt"] = trt.__version__
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "packages": packages,
        "versions": versions,
        "providers": providers,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
    }


def write_benchmark_report(
    path: str | Path,
    benchmarks: Iterable[RuntimeBenchmark],
    extra: dict[str, object] | None = None,
) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "environment": runtime_readiness(),
        "benchmarks": [benchmark.to_dict() for benchmark in benchmarks],
        **(extra or {}),
    }
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return output
