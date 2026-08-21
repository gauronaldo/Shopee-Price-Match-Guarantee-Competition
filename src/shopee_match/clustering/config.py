"""Strict validation-only configuration for Phase 8 entity resolution."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from shopee_match.errors import ConfigurationError
from shopee_match.hashing import matches_frozen_sha256, sha256_file
from shopee_match.retrieval.config import CandidateRetrievalConfig, load_candidate_retrieval_config
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
class EntityResolutionSourceConfig:
    phase7_config_path: Path
    phase7_config_sha256: str
    phase7_metrics_path: Path
    phase7_metrics_sha256: str
    embedding_cache_path: Path
    embedding_cache_sha256: str
    experiment: CandidateRetrievalConfig
    metrics: dict[str, Any]


@dataclass(frozen=True, slots=True)
class PairScoringConfig:
    device: str
    batch_size: int


@dataclass(frozen=True, slots=True)
class GraphSelectionConfig:
    pair_probability_thresholds: tuple[float, ...]
    reciprocal_rank_values: tuple[int, ...]
    cross_component_coverage_values: tuple[float, ...]
    minimum_pairwise_precision: float
    variant_conflict_override_probability: float
    maximum_cluster_size: int
    manual_review_margin: float
    failure_example_limit: int


@dataclass(frozen=True, slots=True)
class EntityResolutionArtifactConfig:
    root: Path
    scored_pairs: Path
    assignments: Path
    metrics: Path
    review: Path
    report: Path


@dataclass(frozen=True, slots=True)
class EntityResolutionConfig:
    seed: int
    source: EntityResolutionSourceConfig
    pair_scoring: PairScoringConfig
    selection: GraphSelectionConfig
    artifacts: EntityResolutionArtifactConfig
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
        if portable_text:
            matches, actual = matches_frozen_sha256(path, expected)
        else:
            actual = sha256_file(path)
            matches = actual == expected
    except OSError as exc:
        raise ConfigurationError(f"Cannot read frozen Phase 8 source: {path}") from exc
    if not matches:
        raise ConfigurationError(
            f"Frozen Phase 8 source hash mismatch for {path}: expected {expected}, got {actual}"
        )
    return path, expected


def load_entity_resolution_config(path: Path) -> EntityResolutionConfig:
    """Load immutable Phase 7 inputs and a validation-only clustering selection protocol."""
    root = _read_yaml(path, "entity resolution config")
    _only_keys(
        root,
        {
            "config_version",
            "seed",
            "source",
            "data",
            "pair_scoring",
            "selection",
            "artifacts",
        },
        "config",
    )
    if root["config_version"] != "phase8.entity_resolution.v1":
        raise ConfigurationError("Unsupported entity resolution config_version")
    seed = _nonnegative_int(root["seed"], "seed")

    source_raw = _mapping(root["source"], "source")
    _only_keys(
        source_raw,
        {
            "phase7_config",
            "phase7_config_sha256",
            "phase7_metrics",
            "phase7_metrics_sha256",
            "embedding_cache",
            "embedding_cache_sha256",
        },
        "source",
    )
    phase7_path, phase7_sha = _verified_file(source_raw, "phase7_config", portable_text=True)
    metrics_path, metrics_sha = _verified_file(source_raw, "phase7_metrics")
    embedding_path, embedding_sha = _verified_file(source_raw, "embedding_cache")
    phase7 = load_candidate_retrieval_config(phase7_path)
    try:
        metrics = cast(dict[str, Any], json.loads(metrics_path.read_text(encoding="utf-8")))
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigurationError("Cannot load frozen Phase 7 metrics") from exc
    if (
        seed != phase7.seed
        or metrics_path != phase7.artifacts.metrics
        or embedding_path != phase7.artifacts.embedding_cache
        or metrics.get("pipeline_version") != "phase7.candidate_retrieval.v1"
        or metrics.get("status") != "phase7_complete_validation_only"
        or metrics.get("data", {}).get("split") != "validation"
        or metrics.get("data", {}).get("test_accessed") is not False
        or metrics.get("selection", {}).get("target_reached") is not True
        or metrics.get("approximate", {}).get("selected_passes") is not True
        or metrics.get("test", {}).get("status") != "disabled_phase7_validation_only"
    ):
        raise ConfigurationError("Phase 8 source is not the accepted validation-only Phase 7 run")
    source = EntityResolutionSourceConfig(
        phase7_config_path=phase7_path,
        phase7_config_sha256=phase7_sha,
        phase7_metrics_path=metrics_path,
        phase7_metrics_sha256=metrics_sha,
        embedding_cache_path=embedding_path,
        embedding_cache_sha256=embedding_sha,
        experiment=phase7,
        metrics=metrics,
    )

    data_raw = _mapping(root["data"], "data")
    _only_keys(data_raw, {"split", "evaluate_test"}, "data")
    if data_raw["split"] != "validation" or data_raw["evaluate_test"] is not False:
        raise ConfigurationError("Phase 8 graph and threshold selection may use validation only")

    pair_raw = _mapping(root["pair_scoring"], "pair_scoring")
    _only_keys(pair_raw, {"device", "batch_size"}, "pair_scoring")
    device = _typed(pair_raw["device"], str, "pair_scoring.device")
    if device not in {"auto", "cpu", "cuda"}:
        raise ConfigurationError("pair_scoring.device must be auto, cpu, or cuda")
    pair_scoring = PairScoringConfig(
        device=device,
        batch_size=_positive_int(pair_raw["batch_size"], "pair_scoring.batch_size"),
    )

    selection_raw = _mapping(root["selection"], "selection")
    _only_keys(
        selection_raw,
        {
            "pair_probability_thresholds",
            "reciprocal_rank_values",
            "cross_component_coverage_values",
            "minimum_pairwise_precision",
            "variant_conflict_override_probability",
            "maximum_cluster_size",
            "manual_review_margin",
            "failure_example_limit",
        },
        "selection",
    )
    ranks_raw = _typed(
        selection_raw["reciprocal_rank_values"], list, "selection.reciprocal_rank_values"
    )
    ranks = tuple(
        _positive_int(value, f"selection.reciprocal_rank_values[{index}]")
        for index, value in enumerate(ranks_raw)
    )
    candidate_k = int(metrics["selection"]["candidate_k"])
    if not ranks or tuple(sorted(set(ranks))) != ranks or max(ranks) > candidate_k:
        raise ConfigurationError("reciprocal ranks must be sorted, unique, and at most candidate K")
    thresholds = _fraction_sequence(
        selection_raw["pair_probability_thresholds"], "selection.pair_probability_thresholds"
    )
    coverage = _fraction_sequence(
        selection_raw["cross_component_coverage_values"],
        "selection.cross_component_coverage_values",
    )
    variant_override = _fraction(
        selection_raw["variant_conflict_override_probability"],
        "selection.variant_conflict_override_probability",
    )
    if variant_override < min(thresholds):
        raise ConfigurationError("variant override probability cannot be below all edge thresholds")
    selection = GraphSelectionConfig(
        pair_probability_thresholds=thresholds,
        reciprocal_rank_values=ranks,
        cross_component_coverage_values=coverage,
        minimum_pairwise_precision=_fraction(
            selection_raw["minimum_pairwise_precision"],
            "selection.minimum_pairwise_precision",
        ),
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
    _only_keys(
        artifact_raw,
        {"root", "scored_pairs", "assignments", "metrics", "review", "report"},
        "artifacts",
    )
    artifacts = EntityResolutionArtifactConfig(
        root=_relative_path(artifact_raw["root"], "artifacts.root"),
        scored_pairs=_relative_path(artifact_raw["scored_pairs"], "artifacts.scored_pairs"),
        assignments=_relative_path(artifact_raw["assignments"], "artifacts.assignments"),
        metrics=_relative_path(artifact_raw["metrics"], "artifacts.metrics"),
        review=_relative_path(artifact_raw["review"], "artifacts.review"),
        report=_relative_path(artifact_raw["report"], "artifacts.report"),
    )
    if any(
        output.parent != artifacts.root
        for output in (
            artifacts.scored_pairs,
            artifacts.assignments,
            artifacts.metrics,
            artifacts.review,
        )
    ):
        raise ConfigurationError("Phase 8 local outputs must live directly under artifacts.root")
    return EntityResolutionConfig(
        seed=seed,
        source=source,
        pair_scoring=pair_scoring,
        selection=selection,
        artifacts=artifacts,
        config_path=path,
    )
