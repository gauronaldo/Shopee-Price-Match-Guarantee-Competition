"""Strict validation-only configuration for entity recall recovery."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from shopee_match.clustering.config import EntityResolutionConfig, load_entity_resolution_config
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
class SingletonAttachmentPolicy:
    """One label-blind second-pass singleton attachment policy."""

    enabled: bool
    probability_threshold: float | None
    reciprocal_rank: int | None
    minimum_support: int
    target_margin: float


@dataclass(frozen=True, slots=True)
class RecoveryAcceptanceConfig:
    minimum_pairwise_precision: float
    minimum_pairwise_recall: float
    minimum_pairwise_f1: float
    minimum_b_cubed_f1: float
    maximum_false_merge_pair_rate: float
    maximum_false_split_group_rate: float


@dataclass(frozen=True, slots=True)
class RecoverySelectionConfig:
    pair_probability_thresholds: tuple[float, ...]
    reciprocal_rank_values: tuple[int, ...]
    cross_component_coverage_values: tuple[float, ...]
    singleton_attachment_policies: tuple[SingletonAttachmentPolicy, ...]
    acceptance: RecoveryAcceptanceConfig
    variant_conflict_override_probability: float
    maximum_cluster_size: int
    manual_review_margin: float
    failure_example_limit: int


@dataclass(frozen=True, slots=True)
class RecoverySourceConfig:
    entity_config_path: Path
    entity_config_sha256: str
    entity_metrics_path: Path
    entity_metrics_sha256: str
    scored_pairs_path: Path
    scored_pairs_sha256: str
    experiment: EntityResolutionConfig
    metrics: dict[str, Any]


@dataclass(frozen=True, slots=True)
class RecoveryArtifactConfig:
    root: Path
    assignments: Path
    metrics: Path
    review: Path
    report: Path


@dataclass(frozen=True, slots=True)
class EntityRecallRecoveryConfig:
    seed: int
    source: RecoverySourceConfig
    selection: RecoverySelectionConfig
    artifacts: RecoveryArtifactConfig
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


def _fraction_sequence(value: Any, location: str) -> tuple[float, ...]:
    raw = _typed(value, list, location)
    result = tuple(_fraction(item, f"{location}[{index}]") for index, item in enumerate(raw))
    if not result or tuple(sorted(set(result))) != result:
        raise ConfigurationError(f"{location} must be non-empty, sorted, and unique")
    return result


def _verified_file(
    raw: dict[str, Any], name: str, *, portable_text: bool = False
) -> tuple[Path, str]:
    path = _relative_path(raw[name], f"source.{name}")
    expected = _digest(raw[f"{name}_sha256"], f"source.{name}_sha256")
    try:
        actual = canonical_text_sha256(path) if portable_text else sha256_file(path)
    except OSError as exc:
        raise ConfigurationError(f"Cannot read frozen recall-recovery source: {path}") from exc
    if actual != expected:
        raise ConfigurationError(
            f"Frozen recall-recovery source hash mismatch for {path}: "
            f"expected {expected}, got {actual}"
        )
    return path, expected


def _load_attachment_policies(
    value: Any, *, candidate_k: int
) -> tuple[SingletonAttachmentPolicy, ...]:
    rows = _typed(value, list, "selection.singleton_attachment_policies")
    policies: list[SingletonAttachmentPolicy] = []
    identities: set[tuple[object, ...]] = set()
    for index, value_row in enumerate(rows):
        location = f"selection.singleton_attachment_policies[{index}]"
        row = _mapping(value_row, location)
        enabled = _typed(row.get("enabled"), bool, f"{location}.enabled")
        if not enabled:
            _only_keys(row, {"enabled"}, location)
            policy = SingletonAttachmentPolicy(False, None, None, 2, 0.0)
        else:
            _only_keys(
                row,
                {
                    "enabled",
                    "probability_threshold",
                    "reciprocal_rank",
                    "minimum_support",
                    "target_margin",
                },
                location,
            )
            rank = _positive_int(row["reciprocal_rank"], f"{location}.reciprocal_rank")
            support = _positive_int(row["minimum_support"], f"{location}.minimum_support")
            if rank > candidate_k:
                raise ConfigurationError(f"{location}.reciprocal_rank cannot exceed candidate K")
            if support < 2:
                raise ConfigurationError(f"{location}.minimum_support must be at least two")
            policy = SingletonAttachmentPolicy(
                True,
                _fraction(row["probability_threshold"], f"{location}.probability_threshold"),
                rank,
                support,
                _fraction(row["target_margin"], f"{location}.target_margin"),
            )
        identity = (
            policy.enabled,
            policy.probability_threshold,
            policy.reciprocal_rank,
            policy.minimum_support,
            policy.target_margin,
        )
        if identity in identities:
            raise ConfigurationError("singleton attachment policies must be unique")
        identities.add(identity)
        policies.append(policy)
    if not policies or not any(not policy.enabled for policy in policies):
        raise ConfigurationError("singleton attachment policies must include a disabled control")
    return tuple(policies)


def load_entity_recall_recovery_config(path: Path) -> EntityRecallRecoveryConfig:
    """Load frozen entity-resolution evidence and a validation-only recovery grid."""
    root = _read_yaml(path, "entity recall-recovery config")
    _only_keys(
        root,
        {"config_version", "seed", "source", "data", "selection", "artifacts"},
        "config",
    )
    if root["config_version"] != "entity_resolution.recall_recovery.v1":
        raise ConfigurationError("Unsupported entity recall-recovery config_version")
    seed = _nonnegative_int(root["seed"], "seed")

    source_raw = _mapping(root["source"], "source")
    _only_keys(
        source_raw,
        {
            "entity_config",
            "entity_config_sha256",
            "entity_metrics",
            "entity_metrics_sha256",
            "scored_pairs",
            "scored_pairs_sha256",
        },
        "source",
    )
    entity_path, entity_sha = _verified_file(source_raw, "entity_config", portable_text=True)
    metrics_path, metrics_sha = _verified_file(source_raw, "entity_metrics")
    pairs_path, pairs_sha = _verified_file(source_raw, "scored_pairs")
    experiment = load_entity_resolution_config(entity_path)
    try:
        metrics = cast(dict[str, Any], json.loads(metrics_path.read_text(encoding="utf-8")))
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigurationError("Cannot load frozen entity-resolution metrics") from exc
    if (
        seed != experiment.seed
        or metrics_path != experiment.artifacts.metrics
        or pairs_path != experiment.artifacts.scored_pairs
        or metrics.get("pipeline_version") != "phase8.entity_resolution.v1"
        or metrics.get("status") != "phase8_complete_validation_only"
        or metrics.get("provenance", {}).get("config_sha256") != entity_sha
        or metrics.get("data", {}).get("split") != "validation"
        or metrics.get("data", {}).get("test_accessed") is not False
        or metrics.get("test", {}).get("status") != "disabled_phase8_validation_only"
    ):
        raise ConfigurationError("Recall recovery requires the accepted validation-only entity run")
    source = RecoverySourceConfig(
        entity_config_path=entity_path,
        entity_config_sha256=entity_sha,
        entity_metrics_path=metrics_path,
        entity_metrics_sha256=metrics_sha,
        scored_pairs_path=pairs_path,
        scored_pairs_sha256=pairs_sha,
        experiment=experiment,
        metrics=metrics,
    )

    data_raw = _mapping(root["data"], "data")
    _only_keys(data_raw, {"split", "evaluate_test"}, "data")
    if data_raw["split"] != "validation" or data_raw["evaluate_test"] is not False:
        raise ConfigurationError("Entity recall recovery may use validation only")

    selection_raw = _mapping(root["selection"], "selection")
    _only_keys(
        selection_raw,
        {
            "pair_probability_thresholds",
            "reciprocal_rank_values",
            "cross_component_coverage_values",
            "singleton_attachment_policies",
            "acceptance",
            "variant_conflict_override_probability",
            "maximum_cluster_size",
            "manual_review_margin",
            "failure_example_limit",
        },
        "selection",
    )
    candidate_k = int(experiment.source.metrics["selection"]["candidate_k"])
    ranks_raw = _typed(
        selection_raw["reciprocal_rank_values"],
        list,
        "selection.reciprocal_rank_values",
    )
    ranks = tuple(
        _positive_int(value, f"selection.reciprocal_rank_values[{index}]")
        for index, value in enumerate(ranks_raw)
    )
    if not ranks or tuple(sorted(set(ranks))) != ranks or max(ranks) > candidate_k:
        raise ConfigurationError("reciprocal ranks must be sorted, unique, and at most candidate K")

    acceptance_raw = _mapping(selection_raw["acceptance"], "selection.acceptance")
    acceptance_keys = {
        "minimum_pairwise_precision",
        "minimum_pairwise_recall",
        "minimum_pairwise_f1",
        "minimum_b_cubed_f1",
        "maximum_false_merge_pair_rate",
        "maximum_false_split_group_rate",
    }
    _only_keys(acceptance_raw, acceptance_keys, "selection.acceptance")
    acceptance = RecoveryAcceptanceConfig(
        **{
            key: _fraction(acceptance_raw[key], f"selection.acceptance.{key}")
            for key in acceptance_keys
        }
    )
    thresholds = _fraction_sequence(
        selection_raw["pair_probability_thresholds"], "selection.pair_probability_thresholds"
    )
    variant_override = _fraction(
        selection_raw["variant_conflict_override_probability"],
        "selection.variant_conflict_override_probability",
    )
    if variant_override < min(thresholds):
        raise ConfigurationError("variant override probability cannot be below all edge thresholds")
    selection = RecoverySelectionConfig(
        pair_probability_thresholds=thresholds,
        reciprocal_rank_values=ranks,
        cross_component_coverage_values=_fraction_sequence(
            selection_raw["cross_component_coverage_values"],
            "selection.cross_component_coverage_values",
        ),
        singleton_attachment_policies=_load_attachment_policies(
            selection_raw["singleton_attachment_policies"], candidate_k=candidate_k
        ),
        acceptance=acceptance,
        variant_conflict_override_probability=variant_override,
        maximum_cluster_size=_positive_int(
            selection_raw["maximum_cluster_size"], "selection.maximum_cluster_size"
        ),
        manual_review_margin=_fraction(
            selection_raw["manual_review_margin"], "selection.manual_review_margin"
        ),
        failure_example_limit=_positive_int(
            selection_raw["failure_example_limit"], "selection.failure_example_limit"
        ),
    )

    artifact_raw = _mapping(root["artifacts"], "artifacts")
    _only_keys(artifact_raw, {"root", "assignments", "metrics", "review", "report"}, "artifacts")
    artifacts = RecoveryArtifactConfig(
        root=_relative_path(artifact_raw["root"], "artifacts.root"),
        assignments=_relative_path(artifact_raw["assignments"], "artifacts.assignments"),
        metrics=_relative_path(artifact_raw["metrics"], "artifacts.metrics"),
        review=_relative_path(artifact_raw["review"], "artifacts.review"),
        report=_relative_path(artifact_raw["report"], "artifacts.report"),
    )
    if any(
        output.parent != artifacts.root
        for output in (artifacts.assignments, artifacts.metrics, artifacts.review, artifacts.report)
    ):
        raise ConfigurationError("Recall-recovery outputs must live directly under artifacts.root")
    return EntityRecallRecoveryConfig(seed, source, selection, artifacts, path)
