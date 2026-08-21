"""SHA-256-locked configuration for one-time final system evaluation."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from shopee_match.clustering.config import EntityResolutionConfig, load_entity_resolution_config
from shopee_match.errors import ConfigurationError
from shopee_match.hashing import matches_frozen_sha256, sha256_file
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


@dataclass(frozen=True, slots=True)
class FinalSourceConfig:
    entity_config_path: Path
    entity_config_sha256: str
    entity_metrics_path: Path
    entity_metrics_sha256: str
    entity_config: EntityResolutionConfig
    entity_metrics: dict[str, Any]


@dataclass(frozen=True, slots=True)
class FrozenSystemPolicy:
    candidate_k: int
    pair_probability_threshold: float
    reciprocal_rank: int
    cross_component_minimum_coverage: float
    variant_conflict_override_probability: float
    maximum_cluster_size: int
    manual_review_margin: float


@dataclass(frozen=True, slots=True)
class FinalRuntimeConfig:
    device: str
    embedding_batch_size: int
    pair_batch_size: int
    num_workers: int


@dataclass(frozen=True, slots=True)
class FinalEvaluationProtocol:
    recall_at: tuple[int, ...]
    average_precision_at: tuple[int, ...]
    exact_block_size: int
    latency_query_count: int
    latency_repetitions: int
    calibration_bins: int
    required_recall: float
    required_precision: float
    failure_example_limit: int


@dataclass(frozen=True, slots=True)
class FinalArtifactConfig:
    root: Path
    access_marker: Path
    embeddings: Path
    scored_pairs: Path
    assignments: Path
    metrics: Path
    review: Path
    report: Path


@dataclass(frozen=True, slots=True)
class FinalSystemEvaluationConfig:
    seed: int
    source: FinalSourceConfig
    policy: FrozenSystemPolicy
    runtime: FinalRuntimeConfig
    evaluation: FinalEvaluationProtocol
    artifacts: FinalArtifactConfig
    config_path: Path


def _digest(value: Any, location: str) -> str:
    result = _typed(value, str, location).lower()
    if len(result) != 64 or any(character not in "0123456789abcdef" for character in result):
        raise ConfigurationError(f"{location} must be a SHA-256 digest")
    return result


def _fraction(value: Any, location: str) -> float:
    result = _number(value, location, allow_zero=True)
    if result > 1:
        raise ConfigurationError(f"{location} must be inside [0, 1]")
    return result


def _verified_file(
    raw: dict[str, Any], name: str, *, portable_text: bool = False
) -> tuple[Path, str]:
    path = _relative_path(raw[name], f"frozen.{name}")
    expected = _digest(raw[f"{name}_sha256"], f"frozen.{name}_sha256")
    try:
        if portable_text:
            matches, actual = matches_frozen_sha256(path, expected)
        else:
            actual = sha256_file(path)
            matches = actual == expected
    except OSError as exc:
        raise ConfigurationError(f"Cannot read frozen final-evaluation source: {path}") from exc
    if not matches:
        raise ConfigurationError(
            f"Frozen final-evaluation source hash mismatch for {path}: "
            f"expected {expected}, got {actual}"
        )
    return path, expected


def _same_number(left: object, right: float | int) -> bool:
    return isinstance(left, int | float) and math.isclose(
        float(left), float(right), rel_tol=0.0, abs_tol=1e-12
    )


def load_final_system_evaluation_config(path: Path) -> FinalSystemEvaluationConfig:
    """Load a protocol that exactly reproduces the validation-frozen system on test."""
    root = _read_yaml(path, "final system evaluation config")
    _only_keys(
        root,
        {"config_version", "seed", "frozen", "data", "runtime", "evaluation", "artifacts"},
        "config",
    )
    if root["config_version"] != "final.system_evaluation.v1":
        raise ConfigurationError("Unsupported final system evaluation config_version")
    seed = _nonnegative_int(root["seed"], "seed")

    frozen_raw = _mapping(root["frozen"], "frozen")
    _only_keys(
        frozen_raw,
        {
            "entity_config",
            "entity_config_sha256",
            "entity_metrics",
            "entity_metrics_sha256",
            "candidate_k",
            "pair_probability_threshold",
            "reciprocal_rank",
            "cross_component_minimum_coverage",
            "variant_conflict_override_probability",
            "maximum_cluster_size",
            "manual_review_margin",
        },
        "frozen",
    )
    entity_path, entity_sha = _verified_file(frozen_raw, "entity_config", portable_text=True)
    metrics_path, metrics_sha = _verified_file(frozen_raw, "entity_metrics")
    entity_config = load_entity_resolution_config(entity_path)
    try:
        entity_metrics = cast(dict[str, Any], json.loads(metrics_path.read_text(encoding="utf-8")))
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigurationError("Cannot load frozen entity-resolution metrics") from exc
    if (
        seed != entity_config.seed
        or metrics_path != entity_config.artifacts.metrics
        or entity_metrics.get("pipeline_version") != "phase8.entity_resolution.v1"
        or entity_metrics.get("status") != "phase8_complete_validation_only"
        or entity_metrics.get("data", {}).get("split") != "validation"
        or entity_metrics.get("data", {}).get("test_accessed") is not False
        or entity_metrics.get("provenance", {}).get("config_sha256") != entity_sha
        or entity_metrics.get("provenance", {}).get("git_dirty") is not False
        or entity_metrics.get("selection", {}).get("selected", {}).get("passes_precision_gate")
        is not True
    ):
        raise ConfigurationError("Final evaluation source is not the clean accepted validation run")
    selected = entity_metrics["selection"]["selected"]
    policy = FrozenSystemPolicy(
        candidate_k=_positive_int(frozen_raw["candidate_k"], "frozen.candidate_k"),
        pair_probability_threshold=_fraction(
            frozen_raw["pair_probability_threshold"], "frozen.pair_probability_threshold"
        ),
        reciprocal_rank=_positive_int(frozen_raw["reciprocal_rank"], "frozen.reciprocal_rank"),
        cross_component_minimum_coverage=_fraction(
            frozen_raw["cross_component_minimum_coverage"],
            "frozen.cross_component_minimum_coverage",
        ),
        variant_conflict_override_probability=_fraction(
            frozen_raw["variant_conflict_override_probability"],
            "frozen.variant_conflict_override_probability",
        ),
        maximum_cluster_size=_positive_int(
            frozen_raw["maximum_cluster_size"], "frozen.maximum_cluster_size"
        ),
        manual_review_margin=_fraction(
            frozen_raw["manual_review_margin"], "frozen.manual_review_margin"
        ),
    )
    if (
        policy.candidate_k != int(entity_metrics["source"]["candidate_k"])
        or not _same_number(
            selected.get("pair_probability_threshold"), policy.pair_probability_threshold
        )
        or selected.get("reciprocal_rank") != policy.reciprocal_rank
        or not _same_number(
            selected.get("cross_component_minimum_coverage"),
            policy.cross_component_minimum_coverage,
        )
        or not _same_number(
            entity_metrics["selection"].get("variant_conflict_override_probability"),
            policy.variant_conflict_override_probability,
        )
        or entity_metrics["selection"].get("maximum_cluster_size") != policy.maximum_cluster_size
        or not math.isclose(
            entity_config.selection.manual_review_margin,
            policy.manual_review_margin,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    ):
        raise ConfigurationError("Explicit final policy differs from validation-selected evidence")
    source = FinalSourceConfig(
        entity_config_path=entity_path,
        entity_config_sha256=entity_sha,
        entity_metrics_path=metrics_path,
        entity_metrics_sha256=metrics_sha,
        entity_config=entity_config,
        entity_metrics=entity_metrics,
    )

    data_raw = _mapping(root["data"], "data")
    _only_keys(data_raw, {"split", "evaluate_once", "allow_test_selection"}, "data")
    if (
        data_raw["split"] != "test"
        or data_raw["evaluate_once"] is not True
        or data_raw["allow_test_selection"] is not False
    ):
        raise ConfigurationError("Final evaluation must be one-time test with no test selection")

    runtime_raw = _mapping(root["runtime"], "runtime")
    _only_keys(
        runtime_raw,
        {"device", "embedding_batch_size", "pair_batch_size", "num_workers"},
        "runtime",
    )
    device = _typed(runtime_raw["device"], str, "runtime.device")
    if device not in {"auto", "cpu", "cuda"}:
        raise ConfigurationError("runtime.device must be auto, cpu, or cuda")
    runtime = FinalRuntimeConfig(
        device=device,
        embedding_batch_size=_positive_int(
            runtime_raw["embedding_batch_size"], "runtime.embedding_batch_size"
        ),
        pair_batch_size=_positive_int(runtime_raw["pair_batch_size"], "runtime.pair_batch_size"),
        num_workers=_nonnegative_int(runtime_raw["num_workers"], "runtime.num_workers"),
    )

    evaluation_raw = _mapping(root["evaluation"], "evaluation")
    _only_keys(
        evaluation_raw,
        {
            "recall_at",
            "average_precision_at",
            "exact_block_size",
            "latency_query_count",
            "latency_repetitions",
            "calibration_bins",
            "required_recall",
            "required_precision",
            "failure_example_limit",
        },
        "evaluation",
    )
    recall_values = _typed(evaluation_raw["recall_at"], list, "evaluation.recall_at")
    recall_at = tuple(
        _positive_int(value, f"evaluation.recall_at[{index}]")
        for index, value in enumerate(recall_values)
    )
    ap_values = _typed(
        evaluation_raw["average_precision_at"], list, "evaluation.average_precision_at"
    )
    average_precision_at = tuple(
        _positive_int(value, f"evaluation.average_precision_at[{index}]")
        for index, value in enumerate(ap_values)
    )
    if (
        not recall_at
        or recall_at != tuple(sorted(set(recall_at)))
        or not average_precision_at
        or average_precision_at != tuple(sorted(set(average_precision_at)))
        or max(*recall_at, *average_precision_at) > policy.candidate_k
    ):
        raise ConfigurationError("Final retrieval K values must be sorted and within candidate K")
    evaluation = FinalEvaluationProtocol(
        recall_at=recall_at,
        average_precision_at=average_precision_at,
        exact_block_size=_positive_int(
            evaluation_raw["exact_block_size"], "evaluation.exact_block_size"
        ),
        latency_query_count=_positive_int(
            evaluation_raw["latency_query_count"], "evaluation.latency_query_count"
        ),
        latency_repetitions=_positive_int(
            evaluation_raw["latency_repetitions"], "evaluation.latency_repetitions"
        ),
        calibration_bins=_positive_int(
            evaluation_raw["calibration_bins"], "evaluation.calibration_bins"
        ),
        required_recall=_fraction(evaluation_raw["required_recall"], "evaluation.required_recall"),
        required_precision=_fraction(
            evaluation_raw["required_precision"], "evaluation.required_precision"
        ),
        failure_example_limit=_positive_int(
            evaluation_raw["failure_example_limit"], "evaluation.failure_example_limit"
        ),
    )

    artifact_raw = _mapping(root["artifacts"], "artifacts")
    _only_keys(
        artifact_raw,
        {
            "root",
            "access_marker",
            "embeddings",
            "scored_pairs",
            "assignments",
            "metrics",
            "review",
            "report",
        },
        "artifacts",
    )
    artifacts = FinalArtifactConfig(
        root=_relative_path(artifact_raw["root"], "artifacts.root"),
        access_marker=_relative_path(artifact_raw["access_marker"], "artifacts.access_marker"),
        embeddings=_relative_path(artifact_raw["embeddings"], "artifacts.embeddings"),
        scored_pairs=_relative_path(artifact_raw["scored_pairs"], "artifacts.scored_pairs"),
        assignments=_relative_path(artifact_raw["assignments"], "artifacts.assignments"),
        metrics=_relative_path(artifact_raw["metrics"], "artifacts.metrics"),
        review=_relative_path(artifact_raw["review"], "artifacts.review"),
        report=_relative_path(artifact_raw["report"], "artifacts.report"),
    )
    if any(
        output.parent != artifacts.root
        for output in (
            artifacts.access_marker,
            artifacts.embeddings,
            artifacts.scored_pairs,
            artifacts.assignments,
            artifacts.metrics,
            artifacts.review,
        )
    ):
        raise ConfigurationError("Final local outputs must live directly under artifacts.root")
    return FinalSystemEvaluationConfig(
        seed=seed,
        source=source,
        policy=policy,
        runtime=runtime,
        evaluation=evaluation,
        artifacts=artifacts,
        config_path=path,
    )
