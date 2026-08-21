"""Strict validation-only configuration for the Phase 9 pretrained benchmark."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import torch

from shopee_match.errors import ConfigurationError
from shopee_match.hashing import matches_frozen_sha256, sha256_file
from shopee_match.retrieval.config import CandidateRetrievalConfig, load_candidate_retrieval_config
from shopee_match.training.text_config import (
    _mapping,
    _nonnegative_int,
    _only_keys,
    _positive_int,
    _read_yaml,
    _relative_path,
    _typed,
)


@dataclass(frozen=True, slots=True)
class PretrainedSourceConfig:
    phase7_config_path: Path
    phase7_config_sha256: str
    phase7_metrics_path: Path
    phase7_metrics_sha256: str
    image_metrics_path: Path
    image_metrics_sha256: str
    weights_path: Path
    weights_sha256: str
    phase7: CandidateRetrievalConfig
    phase7_metrics: dict[str, Any]
    image_metrics: dict[str, Any]


@dataclass(frozen=True, slots=True)
class PretrainedRuntimeConfig:
    device: str
    batch_size: int
    num_workers: int


@dataclass(frozen=True, slots=True)
class PretrainedEvaluationConfig:
    candidate_k: int
    recall_at: tuple[int, ...]
    average_precision_at: tuple[int, ...]
    block_size: int
    latency_query_count: int


@dataclass(frozen=True, slots=True)
class PretrainedArtifactConfig:
    root: Path
    embeddings: Path
    metrics: Path
    review: Path
    report: Path


@dataclass(frozen=True, slots=True)
class PretrainedBenchmarkConfig:
    seed: int
    source: PretrainedSourceConfig
    runtime: PretrainedRuntimeConfig
    evaluation: PretrainedEvaluationConfig
    artifacts: PretrainedArtifactConfig
    config_path: Path


def _digest(value: Any, location: str) -> str:
    result = _typed(value, str, location).lower()
    if len(result) != 64 or any(character not in "0123456789abcdef" for character in result):
        raise ConfigurationError(f"{location} must be a SHA-256 digest")
    return result


def _verified_file(
    raw: dict[str, Any], name: str, *, portable_text: bool = False
) -> tuple[Path, str]:
    path = _relative_path(raw[name], f"source.{name}")
    expected = _digest(raw[f"{name}_sha256"], f"source.{name}_sha256")
    if portable_text:
        matches, actual = matches_frozen_sha256(path, expected)
    else:
        actual = sha256_file(path)
        matches = actual == expected
    if not matches:
        raise ConfigurationError(
            f"Frozen Phase 9 source hash mismatch for {path}: expected {expected}, got {actual}"
        )
    return path, expected


def load_pretrained_benchmark_config(path: Path) -> PretrainedBenchmarkConfig:
    """Load frozen validation sources and official TorchVision weight evidence."""
    root = _read_yaml(path, "pretrained benchmark config")
    _only_keys(
        root,
        {"config_version", "seed", "source", "model", "data", "runtime", "evaluation", "artifacts"},
        "config",
    )
    if root["config_version"] != "phase9.pretrained_efficientnet_b1.v1":
        raise ConfigurationError("Unsupported pretrained benchmark config_version")
    seed = _nonnegative_int(root["seed"], "seed")

    source_raw = _mapping(root["source"], "source")
    _only_keys(
        source_raw,
        {
            "phase7_config",
            "phase7_config_sha256",
            "phase7_metrics",
            "phase7_metrics_sha256",
            "image_metrics",
            "image_metrics_sha256",
        },
        "source",
    )
    phase7_path, phase7_sha = _verified_file(source_raw, "phase7_config", portable_text=True)
    phase7_metrics_path, phase7_metrics_sha = _verified_file(source_raw, "phase7_metrics")
    image_metrics_path, image_metrics_sha = _verified_file(source_raw, "image_metrics")
    phase7 = load_candidate_retrieval_config(phase7_path)
    try:
        phase7_metrics = cast(
            dict[str, Any], json.loads(phase7_metrics_path.read_text(encoding="utf-8"))
        )
        image_metrics = cast(
            dict[str, Any], json.loads(image_metrics_path.read_text(encoding="utf-8"))
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigurationError("Cannot load frozen scratch comparison metrics") from exc
    if (
        seed != phase7.seed
        or phase7_metrics.get("status") != "phase7_complete_validation_only"
        or phase7_metrics.get("data", {}).get("test_accessed") is not False
        or image_metrics.get("pipeline_version") != "phase3.scratch_image_training.v1"
        or image_metrics.get("test", {}).get("status")
        != "disabled_until_checkpoint_and_protocol_are_frozen"
    ):
        raise ConfigurationError("Phase 9 comparison sources are not frozen validation evidence")

    model_raw = _mapping(root["model"], "model")
    _only_keys(
        model_raw,
        {
            "architecture",
            "weights_enum",
            "weights_filename",
            "weights_sha256",
            "weights_source",
            "feature_dimension",
        },
        "model",
    )
    if (
        model_raw["architecture"] != "torchvision_efficientnet_b1"
        or model_raw["weights_enum"] != "IMAGENET1K_V2"
        or model_raw["weights_source"]
        != "https://download.pytorch.org/models/efficientnet_b1-c27df63c.pth"
        or _positive_int(model_raw["feature_dimension"], "model.feature_dimension") != 1280
    ):
        raise ConfigurationError("Phase 9 v1 supports EfficientNet-B1 IMAGENET1K_V2 only")
    filename = _typed(model_raw["weights_filename"], str, "model.weights_filename")
    if Path(filename).name != filename:
        raise ConfigurationError("model.weights_filename must be a portable cache filename")
    weights_path = Path(torch.hub.get_dir()) / "checkpoints" / filename
    weights_sha = _digest(model_raw["weights_sha256"], "model.weights_sha256")
    if sha256_file(weights_path) != weights_sha:
        raise ConfigurationError("Cached pretrained weight SHA-256 does not match the config")
    source = PretrainedSourceConfig(
        phase7_config_path=phase7_path,
        phase7_config_sha256=phase7_sha,
        phase7_metrics_path=phase7_metrics_path,
        phase7_metrics_sha256=phase7_metrics_sha,
        image_metrics_path=image_metrics_path,
        image_metrics_sha256=image_metrics_sha,
        weights_path=weights_path,
        weights_sha256=weights_sha,
        phase7=phase7,
        phase7_metrics=phase7_metrics,
        image_metrics=image_metrics,
    )

    data_raw = _mapping(root["data"], "data")
    _only_keys(data_raw, {"split", "evaluate_test"}, "data")
    if data_raw["split"] != "validation" or data_raw["evaluate_test"] is not False:
        raise ConfigurationError("Phase 9 benchmark may use validation only")

    runtime_raw = _mapping(root["runtime"], "runtime")
    _only_keys(runtime_raw, {"device", "batch_size", "num_workers"}, "runtime")
    device = _typed(runtime_raw["device"], str, "runtime.device")
    if device not in {"auto", "cpu", "cuda"}:
        raise ConfigurationError("runtime.device must be auto, cpu, or cuda")
    runtime = PretrainedRuntimeConfig(
        device=device,
        batch_size=_positive_int(runtime_raw["batch_size"], "runtime.batch_size"),
        num_workers=_nonnegative_int(runtime_raw["num_workers"], "runtime.num_workers"),
    )

    evaluation_raw = _mapping(root["evaluation"], "evaluation")
    _only_keys(
        evaluation_raw,
        {"candidate_k", "recall_at", "average_precision_at", "block_size", "latency_query_count"},
        "evaluation",
    )
    recall_raw = _typed(evaluation_raw["recall_at"], list, "evaluation.recall_at")
    recall_at = tuple(
        _positive_int(value, f"evaluation.recall_at[{index}]")
        for index, value in enumerate(recall_raw)
    )
    ap_raw = _typed(evaluation_raw["average_precision_at"], list, "evaluation.average_precision_at")
    ap_at = tuple(
        _positive_int(value, f"evaluation.average_precision_at[{index}]")
        for index, value in enumerate(ap_raw)
    )
    candidate_k = _positive_int(evaluation_raw["candidate_k"], "evaluation.candidate_k")
    if (
        tuple(sorted(set(recall_at))) != recall_at
        or tuple(sorted(set(ap_at))) != ap_at
        or max(*recall_at, *ap_at) > candidate_k
        or candidate_k != int(phase7_metrics["selection"]["candidate_k"])
    ):
        raise ConfigurationError("Phase 9 K values must match the frozen Phase 7 candidate budget")
    evaluation = PretrainedEvaluationConfig(
        candidate_k=candidate_k,
        recall_at=recall_at,
        average_precision_at=ap_at,
        block_size=_positive_int(evaluation_raw["block_size"], "evaluation.block_size"),
        latency_query_count=_positive_int(
            evaluation_raw["latency_query_count"], "evaluation.latency_query_count"
        ),
    )

    artifact_raw = _mapping(root["artifacts"], "artifacts")
    _only_keys(artifact_raw, {"root", "embeddings", "metrics", "review", "report"}, "artifacts")
    artifacts = PretrainedArtifactConfig(
        root=_relative_path(artifact_raw["root"], "artifacts.root"),
        embeddings=_relative_path(artifact_raw["embeddings"], "artifacts.embeddings"),
        metrics=_relative_path(artifact_raw["metrics"], "artifacts.metrics"),
        review=_relative_path(artifact_raw["review"], "artifacts.review"),
        report=_relative_path(artifact_raw["report"], "artifacts.report"),
    )
    if any(
        output.parent != artifacts.root
        for output in (artifacts.embeddings, artifacts.metrics, artifacts.review)
    ):
        raise ConfigurationError("Phase 9 local outputs must live directly under artifacts.root")
    return PretrainedBenchmarkConfig(seed, source, runtime, evaluation, artifacts, path)
