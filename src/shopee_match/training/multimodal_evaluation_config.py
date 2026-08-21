"""Strict SHA-256-locked configuration for one-time Phase 5 test evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from shopee_match.errors import ConfigurationError
from shopee_match.training.multimodal_config import (
    MultimodalExperimentConfig,
    load_multimodal_experiment_config,
)
from shopee_match.training.text_config import (
    _mapping,
    _nonnegative_int,
    _number,
    _only_keys,
    _positive_int,
    _read_yaml,
    _relative_path,
    _typed,
)
from shopee_match.training.text_evaluation_config import sha256_file


@dataclass(frozen=True, slots=True)
class FrozenMultimodalCheckpointConfig:
    path: Path
    sha256: str
    training_config_path: Path
    training_config_sha256: str
    training_metrics_path: Path
    training_metrics_sha256: str
    validation_metric: str
    validation_metric_value: float
    checkpoint_target: str
    validation_pair_threshold: float
    simple_fusion_image_weight: float


@dataclass(frozen=True, slots=True)
class FrozenMultimodalTestConfig:
    config_version: str
    seed: int
    device: str
    num_workers: int
    batch_size: int
    checkpoint: FrozenMultimodalCheckpointConfig
    training_experiment: MultimodalExperimentConfig
    recall_at: tuple[int, ...]
    average_precision_at: int
    candidate_k: int
    artifact_root: Path
    report_path: Path
    config_path: Path


def _sha256_digest(value: Any, location: str) -> str:
    result = _typed(value, str, location).lower()
    if len(result) != 64 or any(character not in "0123456789abcdef" for character in result):
        raise ConfigurationError(f"{location} must be a lowercase SHA-256 hex digest")
    return result


def load_frozen_multimodal_test_config(path: Path) -> FrozenMultimodalTestConfig:
    """Load immutable Phase 5 artifacts and enforce the validation-frozen protocol."""
    root = _read_yaml(path, "frozen multimodal evaluation config")
    _only_keys(
        root,
        {"config_version", "seed", "runtime", "frozen", "evaluation", "artifacts"},
        "config",
    )
    if root["config_version"] != "phase5.frozen_multimodal_test.v1":
        raise ConfigurationError("Unsupported frozen multimodal evaluation config_version")
    runtime = _mapping(root["runtime"], "runtime")
    _only_keys(runtime, {"device", "num_workers", "batch_size"}, "runtime")
    device = _typed(runtime["device"], str, "runtime.device")
    if device not in {"auto", "cpu", "cuda"}:
        raise ConfigurationError("runtime.device must be auto, cpu, or cuda")

    frozen = _mapping(root["frozen"], "frozen")
    _only_keys(
        frozen,
        {
            "checkpoint",
            "checkpoint_sha256",
            "training_config",
            "training_config_sha256",
            "training_metrics",
            "training_metrics_sha256",
            "validation_metric",
            "validation_metric_value",
            "checkpoint_target",
            "validation_pair_threshold",
            "simple_fusion_image_weight",
        },
        "frozen",
    )
    checkpoint_path = _relative_path(frozen["checkpoint"], "frozen.checkpoint")
    training_config_path = _relative_path(frozen["training_config"], "frozen.training_config")
    metrics_path = _relative_path(frozen["training_metrics"], "frozen.training_metrics")
    checkpoint_sha = _sha256_digest(frozen["checkpoint_sha256"], "frozen.checkpoint_sha256")
    training_config_sha = _sha256_digest(
        frozen["training_config_sha256"], "frozen.training_config_sha256"
    )
    metrics_sha = _sha256_digest(
        frozen["training_metrics_sha256"], "frozen.training_metrics_sha256"
    )
    for input_path, expected, location in (
        (checkpoint_path, checkpoint_sha, "checkpoint"),
        (training_config_path, training_config_sha, "training config"),
        (metrics_path, metrics_sha, "training metrics"),
    ):
        try:
            actual = sha256_file(input_path)
        except OSError as exc:
            raise ConfigurationError(f"Cannot read frozen {location}: {input_path}") from exc
        if actual != expected:
            raise ConfigurationError(
                f"Frozen {location} SHA-256 mismatch: expected {expected}, got {actual}"
            )
    training = load_multimodal_experiment_config(training_config_path)
    validation_metric = _typed(frozen["validation_metric"], str, "frozen.validation_metric")
    checkpoint_target = _typed(frozen["checkpoint_target"], str, "frozen.checkpoint_target")
    if validation_metric != training.evaluation.checkpoint_metric:
        raise ConfigurationError("Frozen validation metric differs from training selection")
    if checkpoint_target != training.evaluation.checkpoint_target:
        raise ConfigurationError("Frozen checkpoint target differs from training selection")
    image_weight = _number(
        frozen["simple_fusion_image_weight"],
        "frozen.simple_fusion_image_weight",
        allow_zero=True,
    )
    if image_weight > 1:
        raise ConfigurationError("simple fusion image weight must be inside [0, 1]")

    evaluation = _mapping(root["evaluation"], "evaluation")
    _only_keys(
        evaluation,
        {
            "split",
            "candidate_pool",
            "exclude_query_itself",
            "recall_at",
            "average_precision_at",
            "candidate_k",
        },
        "evaluation",
    )
    if evaluation["split"] != "test":
        raise ConfigurationError("Frozen multimodal evaluation may target only test")
    if (
        evaluation["candidate_pool"] != "full_split"
        or evaluation["exclude_query_itself"] is not True
    ):
        raise ConfigurationError("Frozen evaluation requires full-split retrieval excluding self")
    recall_raw = _typed(evaluation["recall_at"], list, "evaluation.recall_at")
    recall_at = tuple(
        _positive_int(value, f"evaluation.recall_at[{index}]")
        for index, value in enumerate(recall_raw)
    )
    if not recall_at or recall_at != tuple(sorted(set(recall_at))):
        raise ConfigurationError("evaluation.recall_at must be sorted, unique, and non-empty")
    ap_at = _positive_int(evaluation["average_precision_at"], "evaluation.average_precision_at")
    candidate_k = _positive_int(evaluation["candidate_k"], "evaluation.candidate_k")
    if candidate_k < max(*recall_at, ap_at):
        raise ConfigurationError("candidate_k must cover every configured metric K")
    if (
        recall_at != training.evaluation.recall_at
        or ap_at != training.evaluation.average_precision_at
        or candidate_k != training.evaluation.candidate_k
    ):
        raise ConfigurationError("Frozen test retrieval protocol must match validation exactly")

    artifacts = _mapping(root["artifacts"], "artifacts")
    _only_keys(artifacts, {"root", "report"}, "artifacts")
    return FrozenMultimodalTestConfig(
        config_version=str(root["config_version"]),
        seed=_nonnegative_int(root["seed"], "seed"),
        device=device,
        num_workers=_nonnegative_int(runtime["num_workers"], "runtime.num_workers"),
        batch_size=_positive_int(runtime["batch_size"], "runtime.batch_size"),
        checkpoint=FrozenMultimodalCheckpointConfig(
            path=checkpoint_path,
            sha256=checkpoint_sha,
            training_config_path=training_config_path,
            training_config_sha256=training_config_sha,
            training_metrics_path=metrics_path,
            training_metrics_sha256=metrics_sha,
            validation_metric=validation_metric,
            validation_metric_value=_number(
                frozen["validation_metric_value"], "frozen.validation_metric_value"
            ),
            checkpoint_target=checkpoint_target,
            validation_pair_threshold=_number(
                frozen["validation_pair_threshold"], "frozen.validation_pair_threshold"
            ),
            simple_fusion_image_weight=image_weight,
        ),
        training_experiment=training,
        recall_at=recall_at,
        average_precision_at=ap_at,
        candidate_k=candidate_k,
        artifact_root=_relative_path(artifacts["root"], "artifacts.root"),
        report_path=_relative_path(artifacts["report"], "artifacts.report"),
        config_path=path,
    )
