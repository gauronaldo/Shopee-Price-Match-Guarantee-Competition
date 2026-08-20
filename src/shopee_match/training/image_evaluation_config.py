"""Strict frozen-checkpoint configuration for one-time Phase 3 test evaluation."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from shopee_match.errors import ConfigurationError
from shopee_match.training.image_config import (
    ImageExperimentConfig,
    _mapping,
    _nonnegative_int,
    _only_keys,
    _positive_float,
    _positive_int,
    _read_yaml,
    _relative_path,
    _typed,
    load_image_experiment_config,
)


@dataclass(frozen=True, slots=True)
class FrozenCheckpointConfig:
    path: Path
    sha256: str
    training_config_path: Path
    training_config_sha256: str
    training_metrics_path: Path
    training_metrics_sha256: str
    validation_metric: str
    validation_metric_value: float
    validation_pair_threshold: float


@dataclass(frozen=True, slots=True)
class FrozenImageTestConfig:
    config_version: str
    seed: int
    device: str
    num_workers: int
    batch_size: int
    checkpoint: FrozenCheckpointConfig
    training_experiment: ImageExperimentConfig
    recall_at: tuple[int, ...]
    average_precision_at: int
    candidate_k: int
    artifact_root: Path
    report_path: Path
    config_path: Path


def sha256_file(path: Path) -> str:
    """Hash a frozen input without loading it into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256(value: Any, location: str) -> str:
    result = _typed(value, str, location).lower()
    if len(result) != 64 or any(character not in "0123456789abcdef" for character in result):
        raise ConfigurationError(f"{location} must be a lowercase SHA-256 hex digest")
    return result


def load_frozen_image_test_config(path: Path) -> FrozenImageTestConfig:
    """Load and verify an immutable checkpoint plus validation-frozen test protocol."""
    root = _read_yaml(path, "frozen image evaluation config")
    _only_keys(
        root,
        {"config_version", "seed", "runtime", "frozen", "evaluation", "artifacts"},
        "config",
    )
    if root["config_version"] != "phase3.frozen_image_test.v1":
        raise ConfigurationError("Unsupported frozen image evaluation config_version")

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
            "validation_pair_threshold",
        },
        "frozen",
    )
    checkpoint_path = _relative_path(frozen["checkpoint"], "frozen.checkpoint")
    checkpoint_sha256 = _sha256(frozen["checkpoint_sha256"], "frozen.checkpoint_sha256")
    training_config_path = _relative_path(frozen["training_config"], "frozen.training_config")
    training_config_sha256 = _sha256(
        frozen["training_config_sha256"], "frozen.training_config_sha256"
    )
    training_metrics_path = _relative_path(frozen["training_metrics"], "frozen.training_metrics")
    training_metrics_sha256 = _sha256(
        frozen["training_metrics_sha256"], "frozen.training_metrics_sha256"
    )
    for input_path, expected_hash, location in (
        (checkpoint_path, checkpoint_sha256, "checkpoint"),
        (training_config_path, training_config_sha256, "training config"),
        (training_metrics_path, training_metrics_sha256, "training metrics"),
    ):
        try:
            actual_hash = sha256_file(input_path)
        except OSError as exc:
            raise ConfigurationError(f"Cannot read frozen {location}: {input_path}") from exc
        if actual_hash != expected_hash:
            raise ConfigurationError(
                f"Frozen {location} SHA-256 mismatch: expected {expected_hash}, got {actual_hash}"
            )
    training_experiment = load_image_experiment_config(training_config_path)
    validation_metric = _typed(frozen["validation_metric"], str, "frozen.validation_metric")
    if validation_metric != training_experiment.evaluation.checkpoint_metric:
        raise ConfigurationError("Frozen validation metric differs from training selection metric")

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
        raise ConfigurationError("Frozen image evaluation may target only the test split")
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
    average_precision_at = _positive_int(
        evaluation["average_precision_at"], "evaluation.average_precision_at"
    )
    candidate_k = _positive_int(evaluation["candidate_k"], "evaluation.candidate_k")
    if candidate_k < max(*recall_at, average_precision_at):
        raise ConfigurationError("candidate_k must cover every configured metric K")
    if recall_at != training_experiment.evaluation.recall_at or (
        average_precision_at != training_experiment.evaluation.average_precision_at
        or candidate_k != training_experiment.evaluation.candidate_k
    ):
        raise ConfigurationError("Frozen test retrieval protocol must match validation exactly")

    artifacts = _mapping(root["artifacts"], "artifacts")
    _only_keys(artifacts, {"root", "report"}, "artifacts")
    return FrozenImageTestConfig(
        config_version=str(root["config_version"]),
        seed=_nonnegative_int(root["seed"], "seed"),
        device=device,
        num_workers=_nonnegative_int(runtime["num_workers"], "runtime.num_workers"),
        batch_size=_positive_int(runtime["batch_size"], "runtime.batch_size"),
        checkpoint=FrozenCheckpointConfig(
            path=checkpoint_path,
            sha256=checkpoint_sha256,
            training_config_path=training_config_path,
            training_config_sha256=training_config_sha256,
            training_metrics_path=training_metrics_path,
            training_metrics_sha256=training_metrics_sha256,
            validation_metric=validation_metric,
            validation_metric_value=_positive_float(
                frozen["validation_metric_value"], "frozen.validation_metric_value"
            ),
            validation_pair_threshold=_positive_float(
                frozen["validation_pair_threshold"], "frozen.validation_pair_threshold"
            ),
        ),
        training_experiment=training_experiment,
        recall_at=recall_at,
        average_precision_at=average_precision_at,
        candidate_k=candidate_k,
        artifact_root=_relative_path(artifacts["root"], "artifacts.root"),
        report_path=_relative_path(artifacts["report"], "artifacts.report"),
        config_path=path,
    )
