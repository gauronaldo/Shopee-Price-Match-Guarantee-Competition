"""Frozen configuration for confirmatory evaluation of the hybrid system."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from shopee_match.clustering.hybrid_entity_config import (
    HybridEntityConfig,
    load_hybrid_entity_config,
)
from shopee_match.errors import ConfigurationError
from shopee_match.hashing import canonical_text_sha256, sha256_file
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
class FrozenSingletonAttachment:
    enabled: bool
    probability_threshold: float
    reciprocal_rank: int
    minimum_support: int
    target_margin: float


@dataclass(frozen=True, slots=True)
class FrozenHybridSystemPolicy:
    candidate_k: int
    pair_probability_threshold: float
    reciprocal_rank: int
    cross_component_minimum_coverage: float
    variant_conflict_override_probability: float
    maximum_cluster_size: int
    manual_review_margin: float
    singleton_attachment: FrozenSingletonAttachment


@dataclass(frozen=True, slots=True)
class HybridSystemSource:
    entity_config_path: Path
    entity_config_sha256: str
    entity_metrics_path: Path
    entity_metrics_sha256: str
    entity_config: HybridEntityConfig
    entity_metrics: dict[str, Any]


@dataclass(frozen=True, slots=True)
class HybridSystemRuntime:
    device: str
    embedding_batch_size: int
    pair_batch_size: int
    num_workers: int


@dataclass(frozen=True, slots=True)
class HybridSystemEvaluationProtocol:
    metric_k_values: tuple[int, ...]
    exact_block_size: int
    latency_query_count: int
    latency_repetitions: int
    calibration_bins: int
    required_recall: float
    required_precision: float
    failure_example_limit: int


@dataclass(frozen=True, slots=True)
class HybridSystemArtifacts:
    root: Path
    access_marker: Path
    embeddings: Path
    ranking: Path
    scored_pairs: Path
    assignments: Path
    metrics: Path
    review: Path
    report: Path


@dataclass(frozen=True, slots=True)
class HybridSystemEvaluationConfig:
    seed: int
    source: HybridSystemSource
    policy: FrozenHybridSystemPolicy
    evaluation_manifest_path: Path
    evaluation_manifest_sha256: str
    runtime: HybridSystemRuntime
    evaluation: HybridSystemEvaluationProtocol
    artifacts: HybridSystemArtifacts
    config_path: Path


def _digest(value: Any, location: str) -> str:
    result = _typed(value, str, location).lower()
    if len(result) != 64 or any(character not in "0123456789abcdef" for character in result):
        raise ConfigurationError(f"{location} must be a SHA-256 digest")
    return result


def _fraction(value: Any, location: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ConfigurationError(f"{location} must be numeric")
    result = float(value)
    if not 0.0 <= result <= 1.0:
        raise ConfigurationError(f"{location} must be inside [0, 1]")
    return result


def _same_number(left: object, right: float | int) -> bool:
    return isinstance(left, int | float) and math.isclose(
        float(left), float(right), rel_tol=0.0, abs_tol=1e-12
    )


def _verified_file(
    raw: dict[str, Any], name: str, *, portable_text: bool = False
) -> tuple[Path, str]:
    path = _relative_path(raw[name], f"frozen.{name}")
    expected = _digest(raw[f"{name}_sha256"], f"frozen.{name}_sha256")
    try:
        actual = canonical_text_sha256(path) if portable_text else sha256_file(path)
    except OSError as exc:
        raise ConfigurationError(f"Cannot read frozen hybrid-system source: {path}") from exc
    if actual != expected:
        raise ConfigurationError(
            f"Frozen hybrid-system source mismatch for {path}: expected {expected}, got {actual}"
        )
    return path, expected


def _load_policy(raw: dict[str, Any]) -> FrozenHybridSystemPolicy:
    policy_keys = {
        "candidate_k",
        "pair_probability_threshold",
        "reciprocal_rank",
        "cross_component_minimum_coverage",
        "variant_conflict_override_probability",
        "maximum_cluster_size",
        "manual_review_margin",
        "singleton_attachment",
    }
    _only_keys(raw, policy_keys, "frozen.policy")
    attachment_raw = _mapping(raw["singleton_attachment"], "frozen.policy.singleton_attachment")
    attachment_keys = {
        "enabled",
        "probability_threshold",
        "reciprocal_rank",
        "minimum_support",
        "target_margin",
    }
    _only_keys(attachment_raw, attachment_keys, "frozen.policy.singleton_attachment")
    enabled = _typed(attachment_raw["enabled"], bool, "frozen.policy.singleton_attachment.enabled")
    if not enabled:
        raise ConfigurationError("The accepted hybrid policy requires singleton attachment")
    attachment = FrozenSingletonAttachment(
        enabled=True,
        probability_threshold=_fraction(
            attachment_raw["probability_threshold"],
            "frozen.policy.singleton_attachment.probability_threshold",
        ),
        reciprocal_rank=_positive_int(
            attachment_raw["reciprocal_rank"],
            "frozen.policy.singleton_attachment.reciprocal_rank",
        ),
        minimum_support=_positive_int(
            attachment_raw["minimum_support"],
            "frozen.policy.singleton_attachment.minimum_support",
        ),
        target_margin=_fraction(
            attachment_raw["target_margin"],
            "frozen.policy.singleton_attachment.target_margin",
        ),
    )
    if attachment.minimum_support < 2:
        raise ConfigurationError("Singleton attachment requires at least two supports")
    policy = FrozenHybridSystemPolicy(
        candidate_k=_positive_int(raw["candidate_k"], "frozen.policy.candidate_k"),
        pair_probability_threshold=_fraction(
            raw["pair_probability_threshold"], "frozen.policy.pair_probability_threshold"
        ),
        reciprocal_rank=_positive_int(raw["reciprocal_rank"], "frozen.policy.reciprocal_rank"),
        cross_component_minimum_coverage=_fraction(
            raw["cross_component_minimum_coverage"],
            "frozen.policy.cross_component_minimum_coverage",
        ),
        variant_conflict_override_probability=_fraction(
            raw["variant_conflict_override_probability"],
            "frozen.policy.variant_conflict_override_probability",
        ),
        maximum_cluster_size=_positive_int(
            raw["maximum_cluster_size"], "frozen.policy.maximum_cluster_size"
        ),
        manual_review_margin=_fraction(
            raw["manual_review_margin"], "frozen.policy.manual_review_margin"
        ),
        singleton_attachment=attachment,
    )
    if attachment.reciprocal_rank > policy.candidate_k:
        raise ConfigurationError("Singleton attachment rank cannot exceed candidate K")
    return policy


def load_hybrid_system_evaluation_config(path: Path) -> HybridSystemEvaluationConfig:
    """Load the validation-frozen hybrid policy without accessing test rows."""
    root = _read_yaml(path, "hybrid system evaluation config")
    _only_keys(
        root,
        {"config_version", "seed", "frozen", "data", "runtime", "evaluation", "artifacts"},
        "config",
    )
    if root["config_version"] != "hybrid.system_evaluation.v1":
        raise ConfigurationError("Unsupported hybrid system evaluation config_version")
    seed = _nonnegative_int(root["seed"], "seed")

    frozen_raw = _mapping(root["frozen"], "frozen")
    _only_keys(
        frozen_raw,
        {
            "entity_config",
            "entity_config_sha256",
            "entity_metrics",
            "entity_metrics_sha256",
            "policy",
        },
        "frozen",
    )
    entity_path, entity_sha = _verified_file(frozen_raw, "entity_config", portable_text=True)
    metrics_path, metrics_sha = _verified_file(frozen_raw, "entity_metrics")
    entity_config = load_hybrid_entity_config(entity_path)
    try:
        entity_metrics = cast(dict[str, Any], json.loads(metrics_path.read_text(encoding="utf-8")))
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigurationError("Cannot load frozen hybrid entity metrics") from exc
    if (
        seed != entity_config.seed
        or metrics_path != entity_config.artifacts.metrics
        or entity_metrics.get("pipeline_version") != "hybrid_entity_resolution.v1"
        or entity_metrics.get("status") != "hybrid_entity_target_reached_validation_only"
        or entity_metrics.get("data", {}).get("split") != "validation"
        or entity_metrics.get("data", {}).get("test_accessed") is not False
        or entity_metrics.get("provenance", {}).get("config_sha256") != entity_sha
        or entity_metrics.get("provenance", {}).get("git_dirty") is not False
        or entity_config.source.hybrid_metrics.get("provenance", {}).get("git_dirty") is not False
        or entity_metrics.get("selection", {}).get("selected", {}).get("passes_quality_target")
        is not True
    ):
        raise ConfigurationError(
            "Hybrid final evaluation requires clean accepted validation evidence"
        )

    policy = _load_policy(_mapping(frozen_raw["policy"], "frozen.policy"))
    selected = entity_metrics["selection"]["selected"]
    selected_attachment = selected.get("singleton_attachment", {})
    selection = entity_config.source.recovery.selection
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
            selection.variant_conflict_override_probability,
            policy.variant_conflict_override_probability,
        )
        or selection.maximum_cluster_size != policy.maximum_cluster_size
        or not _same_number(selection.manual_review_margin, policy.manual_review_margin)
        or selected_attachment.get("enabled") is not True
        or not _same_number(
            selected_attachment.get("probability_threshold"),
            policy.singleton_attachment.probability_threshold,
        )
        or selected_attachment.get("reciprocal_rank") != policy.singleton_attachment.reciprocal_rank
        or selected_attachment.get("minimum_support") != policy.singleton_attachment.minimum_support
        or not _same_number(
            selected_attachment.get("target_margin"), policy.singleton_attachment.target_margin
        )
    ):
        raise ConfigurationError("Explicit final policy differs from validation-selected evidence")
    source = HybridSystemSource(
        entity_path,
        entity_sha,
        metrics_path,
        metrics_sha,
        entity_config,
        entity_metrics,
    )

    data_raw = _mapping(root["data"], "data")
    _only_keys(
        data_raw,
        {
            "split",
            "evaluation_manifest",
            "evaluation_manifest_sha256",
            "evaluate_once",
            "allow_test_selection",
        },
        "data",
    )
    if (
        data_raw["split"] != "test"
        or data_raw["evaluate_once"] is not True
        or data_raw["allow_test_selection"] is not False
    ):
        raise ConfigurationError("Hybrid system evaluation must be one-time test without selection")
    evaluation_manifest = _relative_path(
        data_raw["evaluation_manifest"], "data.evaluation_manifest"
    )
    evaluation_manifest_sha = _digest(
        data_raw["evaluation_manifest_sha256"], "data.evaluation_manifest_sha256"
    )
    try:
        actual_manifest_sha = sha256_file(evaluation_manifest)
    except OSError as exc:
        raise ConfigurationError(
            f"Cannot read frozen confirmation manifest: {evaluation_manifest}"
        ) from exc
    if actual_manifest_sha != evaluation_manifest_sha:
        raise ConfigurationError(
            "Frozen confirmation manifest hash differs from the evaluation contract"
        )
    training_manifest = (
        entity_config.source.hybrid.source.experiment.source.experiment.source.experiment.data.split_manifest
    )
    if evaluation_manifest == training_manifest:
        raise ConfigurationError(
            "Confirmation evaluation must not reuse the training compatibility manifest"
        )

    runtime_raw = _mapping(root["runtime"], "runtime")
    _only_keys(
        runtime_raw,
        {"device", "embedding_batch_size", "pair_batch_size", "num_workers"},
        "runtime",
    )
    device = _typed(runtime_raw["device"], str, "runtime.device")
    if device not in {"auto", "cpu", "cuda"}:
        raise ConfigurationError("runtime.device must be auto, cpu, or cuda")
    runtime = HybridSystemRuntime(
        device,
        _positive_int(runtime_raw["embedding_batch_size"], "runtime.embedding_batch_size"),
        _positive_int(runtime_raw["pair_batch_size"], "runtime.pair_batch_size"),
        _nonnegative_int(runtime_raw["num_workers"], "runtime.num_workers"),
    )

    evaluation_raw = _mapping(root["evaluation"], "evaluation")
    evaluation_keys = {
        "metric_k_values",
        "exact_block_size",
        "latency_query_count",
        "latency_repetitions",
        "calibration_bins",
        "required_recall",
        "required_precision",
        "failure_example_limit",
    }
    _only_keys(evaluation_raw, evaluation_keys, "evaluation")
    raw_k = _typed(evaluation_raw["metric_k_values"], list, "evaluation.metric_k_values")
    k_values = tuple(
        _positive_int(value, f"evaluation.metric_k_values[{index}]")
        for index, value in enumerate(raw_k)
    )
    if (
        not k_values
        or tuple(sorted(set(k_values))) != k_values
        or max(k_values) > policy.candidate_k
    ):
        raise ConfigurationError("Metric K values must be sorted, unique, and within candidate K")
    evaluation = HybridSystemEvaluationProtocol(
        k_values,
        _positive_int(evaluation_raw["exact_block_size"], "evaluation.exact_block_size"),
        _positive_int(evaluation_raw["latency_query_count"], "evaluation.latency_query_count"),
        _positive_int(evaluation_raw["latency_repetitions"], "evaluation.latency_repetitions"),
        _positive_int(evaluation_raw["calibration_bins"], "evaluation.calibration_bins"),
        _fraction(evaluation_raw["required_recall"], "evaluation.required_recall"),
        _fraction(evaluation_raw["required_precision"], "evaluation.required_precision"),
        _positive_int(evaluation_raw["failure_example_limit"], "evaluation.failure_example_limit"),
    )

    artifacts_raw = _mapping(root["artifacts"], "artifacts")
    artifact_names = {
        "root",
        "access_marker",
        "embeddings",
        "ranking",
        "scored_pairs",
        "assignments",
        "metrics",
        "review",
        "report",
    }
    _only_keys(artifacts_raw, artifact_names, "artifacts")
    artifacts = HybridSystemArtifacts(
        **{
            name: _relative_path(artifacts_raw[name], f"artifacts.{name}")
            for name in artifact_names
        }
    )
    if any(
        output.parent != artifacts.root
        for output in (
            artifacts.access_marker,
            artifacts.embeddings,
            artifacts.ranking,
            artifacts.scored_pairs,
            artifacts.assignments,
            artifacts.metrics,
            artifacts.review,
            artifacts.report,
        )
    ):
        raise ConfigurationError("Hybrid final outputs must live directly under artifacts.root")
    return HybridSystemEvaluationConfig(
        seed,
        source,
        policy,
        evaluation_manifest,
        evaluation_manifest_sha,
        runtime,
        evaluation,
        artifacts,
        path,
    )
