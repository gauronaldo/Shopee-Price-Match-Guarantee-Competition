"""Frozen validation-only graph selection for recalibrated pair-evidence scores."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from shopee_match.clustering.recovery_config import (
    RecoveryAcceptanceConfig,
    RecoverySelectionConfig,
    _fraction,
    _fraction_sequence,
    _load_attachment_policies,
)
from shopee_match.errors import ConfigurationError
from shopee_match.hashing import canonical_text_sha256, sha256_file
from shopee_match.training.pair_evidence_config import (
    PairEvidenceExperimentConfig,
    load_pair_evidence_experiment_config,
)
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
class PairEvidenceGraphSourceConfig:
    evidence_config_path: Path
    evidence_config_sha256: str
    evidence_metrics_path: Path
    evidence_metrics_sha256: str
    checkpoint_path: Path
    checkpoint_sha256: str
    scored_pairs_path: Path
    scored_pairs_sha256: str
    experiment: PairEvidenceExperimentConfig
    metrics: dict[str, Any]


@dataclass(frozen=True, slots=True)
class PairEvidenceGraphArtifactConfig:
    root: Path
    assignments: Path
    metrics: Path
    review: Path
    report: Path


@dataclass(frozen=True, slots=True)
class PairEvidenceGraphConfig:
    seed: int
    source: PairEvidenceGraphSourceConfig
    selection: RecoverySelectionConfig
    artifacts: PairEvidenceGraphArtifactConfig
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
    try:
        actual = canonical_text_sha256(path) if portable_text else sha256_file(path)
    except OSError as exc:
        raise ConfigurationError(f"Cannot read frozen pair-evidence graph source: {path}") from exc
    if actual != expected:
        raise ConfigurationError(
            f"Frozen pair-evidence graph source mismatch for {path}: "
            f"expected {expected}, got {actual}"
        )
    return path, expected


def _load_selection(raw: dict[str, Any], *, candidate_k: int) -> RecoverySelectionConfig:
    selection_keys = {
        "pair_probability_thresholds",
        "reciprocal_rank_values",
        "cross_component_coverage_values",
        "singleton_attachment_policies",
        "acceptance",
        "variant_conflict_override_probability",
        "maximum_cluster_size",
        "manual_review_margin",
        "failure_example_limit",
    }
    _only_keys(raw, selection_keys, "selection")
    ranks_raw = _typed(raw["reciprocal_rank_values"], list, "selection.reciprocal_rank_values")
    ranks = tuple(
        _positive_int(value, f"selection.reciprocal_rank_values[{index}]")
        for index, value in enumerate(ranks_raw)
    )
    if not ranks or tuple(sorted(set(ranks))) != ranks or max(ranks) > candidate_k:
        raise ConfigurationError("reciprocal ranks must be sorted, unique, and at most candidate K")
    acceptance_raw = _mapping(raw["acceptance"], "selection.acceptance")
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
        minimum_pairwise_precision=_fraction(
            acceptance_raw["minimum_pairwise_precision"],
            "selection.acceptance.minimum_pairwise_precision",
        ),
        minimum_pairwise_recall=_fraction(
            acceptance_raw["minimum_pairwise_recall"],
            "selection.acceptance.minimum_pairwise_recall",
        ),
        minimum_pairwise_f1=_fraction(
            acceptance_raw["minimum_pairwise_f1"],
            "selection.acceptance.minimum_pairwise_f1",
        ),
        minimum_b_cubed_f1=_fraction(
            acceptance_raw["minimum_b_cubed_f1"],
            "selection.acceptance.minimum_b_cubed_f1",
        ),
        maximum_false_merge_pair_rate=_fraction(
            acceptance_raw["maximum_false_merge_pair_rate"],
            "selection.acceptance.maximum_false_merge_pair_rate",
        ),
        maximum_false_split_group_rate=_fraction(
            acceptance_raw["maximum_false_split_group_rate"],
            "selection.acceptance.maximum_false_split_group_rate",
        ),
    )
    thresholds = _fraction_sequence(
        raw["pair_probability_thresholds"], "selection.pair_probability_thresholds"
    )
    variant_override = _fraction(
        raw["variant_conflict_override_probability"],
        "selection.variant_conflict_override_probability",
    )
    if variant_override < min(thresholds):
        raise ConfigurationError("variant override probability cannot be below all edge thresholds")
    return RecoverySelectionConfig(
        pair_probability_thresholds=thresholds,
        reciprocal_rank_values=ranks,
        cross_component_coverage_values=_fraction_sequence(
            raw["cross_component_coverage_values"],
            "selection.cross_component_coverage_values",
        ),
        singleton_attachment_policies=_load_attachment_policies(
            raw["singleton_attachment_policies"], candidate_k=candidate_k
        ),
        acceptance=acceptance,
        variant_conflict_override_probability=variant_override,
        maximum_cluster_size=_positive_int(
            raw["maximum_cluster_size"], "selection.maximum_cluster_size"
        ),
        manual_review_margin=_fraction(
            raw["manual_review_margin"], "selection.manual_review_margin"
        ),
        failure_example_limit=_positive_int(
            raw["failure_example_limit"], "selection.failure_example_limit"
        ),
    )


def load_pair_evidence_graph_config(path: Path) -> PairEvidenceGraphConfig:
    root = _read_yaml(path, "pair-evidence graph config")
    _only_keys(
        root,
        {"config_version", "seed", "source", "data", "selection", "artifacts"},
        "config",
    )
    if root["config_version"] != "pair_evidence.graph_selection.v1":
        raise ConfigurationError("Unsupported pair-evidence graph config_version")
    seed = _nonnegative_int(root["seed"], "seed")
    source_raw = _mapping(root["source"], "source")
    source_names = {"evidence_config", "evidence_metrics", "checkpoint", "scored_pairs"}
    _only_keys(
        source_raw,
        source_names | {f"{name}_sha256" for name in source_names},
        "source",
    )
    evidence_path, evidence_sha = _verified_file(source_raw, "evidence_config", portable_text=True)
    metrics_path, metrics_sha = _verified_file(source_raw, "evidence_metrics")
    checkpoint_path, checkpoint_sha = _verified_file(source_raw, "checkpoint")
    pairs_path, pairs_sha = _verified_file(source_raw, "scored_pairs")
    experiment = load_pair_evidence_experiment_config(evidence_path)
    try:
        metrics = cast(dict[str, Any], json.loads(metrics_path.read_text(encoding="utf-8")))
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigurationError("Cannot load frozen pair-evidence metrics") from exc
    if (
        seed != experiment.seed
        or metrics_path != experiment.artifacts.metrics
        or checkpoint_path != experiment.artifacts.checkpoint
        or pairs_path != experiment.artifacts.rescored_pairs
        or metrics.get("pipeline_version") != "pair_evidence.training.v1"
        or metrics.get("provenance", {}).get("config_sha256") != evidence_sha
        or metrics.get("data", {}).get("test_accessed") is not False
        or metrics.get("test", {}).get("status") != "disabled_pair_evidence_validation_only"
    ):
        raise ConfigurationError(
            "Graph selection requires the accepted validation-only evidence run"
        )
    source = PairEvidenceGraphSourceConfig(
        evidence_path,
        evidence_sha,
        metrics_path,
        metrics_sha,
        checkpoint_path,
        checkpoint_sha,
        pairs_path,
        pairs_sha,
        experiment,
        metrics,
    )
    data_raw = _mapping(root["data"], "data")
    _only_keys(data_raw, {"split", "evaluate_test", "retrain_model"}, "data")
    if (
        data_raw["split"] != "validation"
        or data_raw["evaluate_test"] is not False
        or data_raw["retrain_model"] is not False
    ):
        raise ConfigurationError("Pair-evidence graph selection is validation-only and score-only")
    candidate_k = int(
        experiment.recovery.source.experiment.source.metrics["selection"]["candidate_k"]
    )
    selection = _load_selection(_mapping(root["selection"], "selection"), candidate_k=candidate_k)
    artifact_raw = _mapping(root["artifacts"], "artifacts")
    _only_keys(artifact_raw, {"root", "assignments", "metrics", "review", "report"}, "artifacts")
    artifacts = PairEvidenceGraphArtifactConfig(
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
        raise ConfigurationError("Graph-selection outputs must live directly under artifacts.root")
    return PairEvidenceGraphConfig(seed, source, selection, artifacts, path)
