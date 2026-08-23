"""Frozen configuration for downstream evaluation of hybrid candidates."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from shopee_match.clustering.recovery_config import (
    EntityRecallRecoveryConfig,
    load_entity_recall_recovery_config,
)
from shopee_match.errors import ConfigurationError
from shopee_match.hashing import canonical_text_sha256, sha256_file
from shopee_match.retrieval.hybrid_config import HybridRetrievalConfig, load_hybrid_retrieval_config
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
class HybridEntitySourceConfig:
    hybrid_config_path: Path
    hybrid_config_sha256: str
    hybrid_metrics_path: Path
    hybrid_metrics_sha256: str
    hybrid_ranking_path: Path
    hybrid_ranking_sha256: str
    recovery_config_path: Path
    recovery_config_sha256: str
    hybrid: HybridRetrievalConfig
    hybrid_metrics: dict[str, Any]
    recovery: EntityRecallRecoveryConfig


@dataclass(frozen=True, slots=True)
class HybridEntityArtifactConfig:
    root: Path
    scored_pairs: Path
    assignments: Path
    metrics: Path
    review: Path
    report: Path


@dataclass(frozen=True, slots=True)
class HybridEntityConfig:
    seed: int
    source: HybridEntitySourceConfig
    device: str
    pair_batch_size: int
    artifacts: HybridEntityArtifactConfig
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
        raise ConfigurationError(f"Cannot read frozen hybrid-entity source: {path}") from exc
    if actual != expected:
        raise ConfigurationError(
            f"Frozen hybrid-entity source mismatch for {path}: expected {expected}, got {actual}"
        )
    return path, expected


def load_hybrid_entity_config(path: Path) -> HybridEntityConfig:
    root = _read_yaml(path, "hybrid entity config")
    _only_keys(
        root,
        {"config_version", "seed", "source", "data", "pair_scoring", "artifacts"},
        "config",
    )
    if root["config_version"] != "hybrid_entity_resolution.v1":
        raise ConfigurationError("Unsupported hybrid entity config_version")
    seed = _nonnegative_int(root["seed"], "seed")
    source_raw = _mapping(root["source"], "source")
    source_names = {
        "hybrid_config",
        "hybrid_metrics",
        "hybrid_ranking",
        "recovery_config",
    }
    _only_keys(
        source_raw,
        source_names | {f"{name}_sha256" for name in source_names},
        "source",
    )
    hybrid_path, hybrid_sha = _verified_file(source_raw, "hybrid_config", portable_text=True)
    metrics_path, metrics_sha = _verified_file(source_raw, "hybrid_metrics")
    ranking_path, ranking_sha = _verified_file(source_raw, "hybrid_ranking")
    recovery_path, recovery_sha = _verified_file(source_raw, "recovery_config", portable_text=True)
    hybrid = load_hybrid_retrieval_config(hybrid_path)
    recovery = load_entity_recall_recovery_config(recovery_path)
    try:
        metrics = cast(dict[str, Any], json.loads(metrics_path.read_text(encoding="utf-8")))
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigurationError("Cannot load frozen hybrid retrieval metrics") from exc
    if (
        seed != hybrid.seed
        or seed != recovery.seed
        or metrics_path != hybrid.artifacts.metrics
        or ranking_path != hybrid.artifacts.ranking
        or metrics.get("pipeline_version") != "hybrid_candidate_retrieval.v1"
        or metrics.get("status") != "hybrid_candidate_target_reached_validation_only"
        or metrics.get("selection", {}).get("target_reached") is not True
        or metrics.get("data", {}).get("test_accessed") is not False
        or metrics.get("test", {}).get("status") != "disabled_hybrid_retrieval_validation_only"
        or hybrid.source.phase7_config_sha256
        != recovery.source.experiment.source.phase7_config_sha256
    ):
        raise ConfigurationError("Hybrid entity evaluation requires aligned validation-only inputs")
    source = HybridEntitySourceConfig(
        hybrid_path,
        hybrid_sha,
        metrics_path,
        metrics_sha,
        ranking_path,
        ranking_sha,
        recovery_path,
        recovery_sha,
        hybrid,
        metrics,
        recovery,
    )
    data_raw = _mapping(root["data"], "data")
    _only_keys(data_raw, {"split", "evaluate_test", "retrain_model"}, "data")
    if (
        data_raw["split"] != "validation"
        or data_raw["evaluate_test"] is not False
        or data_raw["retrain_model"] is not False
    ):
        raise ConfigurationError("Hybrid entity evaluation is validation-only and score-only")
    pair_raw = _mapping(root["pair_scoring"], "pair_scoring")
    _only_keys(pair_raw, {"device", "batch_size"}, "pair_scoring")
    device = _typed(pair_raw["device"], str, "pair_scoring.device")
    if device not in {"auto", "cpu", "cuda"}:
        raise ConfigurationError("pair_scoring.device must be auto, cpu, or cuda")
    artifact_raw = _mapping(root["artifacts"], "artifacts")
    artifact_names = {"root", "scored_pairs", "assignments", "metrics", "review", "report"}
    _only_keys(artifact_raw, artifact_names, "artifacts")
    artifacts = HybridEntityArtifactConfig(
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
            artifacts.report,
        )
    ):
        raise ConfigurationError("Hybrid entity outputs must live directly under artifacts.root")
    return HybridEntityConfig(
        seed,
        source,
        device,
        _positive_int(pair_raw["batch_size"], "pair_scoring.batch_size"),
        artifacts,
        path,
    )
