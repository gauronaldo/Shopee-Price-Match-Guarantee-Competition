"""Strict validation-only configuration for hybrid candidate retrieval."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from shopee_match.errors import ConfigurationError
from shopee_match.hashing import canonical_text_sha256, sha256_file
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
class HybridSourceConfig:
    phase7_config_path: Path
    phase7_config_sha256: str
    phase7_metrics_path: Path
    phase7_metrics_sha256: str
    embedding_cache_path: Path
    embedding_cache_sha256: str
    experiment: CandidateRetrievalConfig
    metrics: dict[str, Any]


@dataclass(frozen=True, slots=True)
class HybridTfidfConfig:
    ngram_range: tuple[int, int]
    max_features: int
    candidate_k: int


@dataclass(frozen=True, slots=True)
class HybridFusionConfig:
    dense_candidate_k: int
    phash_candidate_k: int
    rrf_constant: int
    dense_weight: float
    tfidf_weight: float
    phash_weight: float
    evaluation_k_values: tuple[int, ...]
    target_recall: float


@dataclass(frozen=True, slots=True)
class HybridArtifactConfig:
    root: Path
    ranking: Path
    metrics: Path
    report: Path


@dataclass(frozen=True, slots=True)
class HybridRetrievalConfig:
    seed: int
    source: HybridSourceConfig
    tfidf: HybridTfidfConfig
    fusion: HybridFusionConfig
    artifacts: HybridArtifactConfig
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
        raise ConfigurationError(f"Cannot read frozen hybrid-retrieval source: {path}") from exc
    if actual != expected:
        raise ConfigurationError(
            f"Frozen hybrid-retrieval source mismatch for {path}: expected {expected}, got {actual}"
        )
    return path, expected


def _positive_number(value: Any, location: str) -> float:
    return _number(value, location, allow_zero=False)


def load_hybrid_retrieval_config(path: Path) -> HybridRetrievalConfig:
    root = _read_yaml(path, "hybrid retrieval config")
    _only_keys(
        root,
        {"config_version", "seed", "source", "data", "tfidf", "fusion", "artifacts"},
        "config",
    )
    if root["config_version"] != "hybrid_candidate_retrieval.v1":
        raise ConfigurationError("Unsupported hybrid retrieval config_version")
    seed = _nonnegative_int(root["seed"], "seed")
    source_raw = _mapping(root["source"], "source")
    source_names = {"phase7_config", "phase7_metrics", "embedding_cache"}
    _only_keys(
        source_raw,
        source_names | {f"{name}_sha256" for name in source_names},
        "source",
    )
    phase7_path, phase7_sha = _verified_file(source_raw, "phase7_config", portable_text=True)
    metrics_path, metrics_sha = _verified_file(source_raw, "phase7_metrics")
    embedding_path, embedding_sha = _verified_file(source_raw, "embedding_cache")
    experiment = load_candidate_retrieval_config(phase7_path)
    try:
        metrics = cast(dict[str, Any], json.loads(metrics_path.read_text(encoding="utf-8")))
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigurationError("Cannot load frozen Phase 7 metrics") from exc
    if (
        seed != experiment.seed
        or metrics_path != experiment.artifacts.metrics
        or embedding_path != experiment.artifacts.embedding_cache
        or metrics.get("pipeline_version") != "phase7.candidate_retrieval.v1"
        or metrics.get("status") != "phase7_complete_validation_only"
        or metrics.get("data", {}).get("test_accessed") is not False
        or metrics.get("test", {}).get("status") != "disabled_phase7_validation_only"
    ):
        raise ConfigurationError(
            "Hybrid retrieval requires the accepted validation-only Phase 7 run"
        )
    source = HybridSourceConfig(
        phase7_path,
        phase7_sha,
        metrics_path,
        metrics_sha,
        embedding_path,
        embedding_sha,
        experiment,
        metrics,
    )
    data_raw = _mapping(root["data"], "data")
    _only_keys(data_raw, {"fit_split", "evaluation_split", "evaluate_test"}, "data")
    if (
        data_raw["fit_split"] != "train"
        or data_raw["evaluation_split"] != "validation"
        or data_raw["evaluate_test"] is not False
    ):
        raise ConfigurationError("Hybrid retrieval must fit on train and evaluate validation only")

    tfidf_raw = _mapping(root["tfidf"], "tfidf")
    _only_keys(tfidf_raw, {"ngram_range", "max_features", "candidate_k"}, "tfidf")
    ngram_raw = _typed(tfidf_raw["ngram_range"], list, "tfidf.ngram_range")
    if len(ngram_raw) != 2:
        raise ConfigurationError("tfidf.ngram_range must contain two integers")
    ngram_range = (
        _positive_int(ngram_raw[0], "tfidf.ngram_range[0]"),
        _positive_int(ngram_raw[1], "tfidf.ngram_range[1]"),
    )
    if ngram_range[0] > ngram_range[1]:
        raise ConfigurationError("tfidf.ngram_range must be increasing")
    tfidf = HybridTfidfConfig(
        ngram_range,
        _positive_int(tfidf_raw["max_features"], "tfidf.max_features"),
        _positive_int(tfidf_raw["candidate_k"], "tfidf.candidate_k"),
    )

    fusion_raw = _mapping(root["fusion"], "fusion")
    fusion_keys = {
        "dense_candidate_k",
        "phash_candidate_k",
        "rrf_constant",
        "dense_weight",
        "tfidf_weight",
        "phash_weight",
        "evaluation_k_values",
        "target_recall",
    }
    _only_keys(fusion_raw, fusion_keys, "fusion")
    k_raw = _typed(fusion_raw["evaluation_k_values"], list, "fusion.evaluation_k_values")
    k_values = tuple(
        _positive_int(value, f"fusion.evaluation_k_values[{index}]")
        for index, value in enumerate(k_raw)
    )
    maximum_union = (
        _positive_int(fusion_raw["dense_candidate_k"], "fusion.dense_candidate_k")
        + tfidf.candidate_k
        + _positive_int(fusion_raw["phash_candidate_k"], "fusion.phash_candidate_k")
    )
    if not k_values or tuple(sorted(set(k_values))) != k_values or max(k_values) > maximum_union:
        raise ConfigurationError(
            "evaluation K values must be sorted, unique, and within union size"
        )
    target_recall = _number(fusion_raw["target_recall"], "fusion.target_recall", allow_zero=False)
    if target_recall > 1:
        raise ConfigurationError("fusion.target_recall must be inside (0, 1]")
    fusion = HybridFusionConfig(
        dense_candidate_k=_positive_int(
            fusion_raw["dense_candidate_k"], "fusion.dense_candidate_k"
        ),
        phash_candidate_k=_positive_int(
            fusion_raw["phash_candidate_k"], "fusion.phash_candidate_k"
        ),
        rrf_constant=_positive_int(fusion_raw["rrf_constant"], "fusion.rrf_constant"),
        dense_weight=_positive_number(fusion_raw["dense_weight"], "fusion.dense_weight"),
        tfidf_weight=_positive_number(fusion_raw["tfidf_weight"], "fusion.tfidf_weight"),
        phash_weight=_positive_number(fusion_raw["phash_weight"], "fusion.phash_weight"),
        evaluation_k_values=k_values,
        target_recall=float(target_recall),
    )
    artifact_raw = _mapping(root["artifacts"], "artifacts")
    _only_keys(artifact_raw, {"root", "ranking", "metrics", "report"}, "artifacts")
    artifacts = HybridArtifactConfig(
        root=_relative_path(artifact_raw["root"], "artifacts.root"),
        ranking=_relative_path(artifact_raw["ranking"], "artifacts.ranking"),
        metrics=_relative_path(artifact_raw["metrics"], "artifacts.metrics"),
        report=_relative_path(artifact_raw["report"], "artifacts.report"),
    )
    if any(
        output.parent != artifacts.root
        for output in (artifacts.ranking, artifacts.metrics, artifacts.report)
    ):
        raise ConfigurationError("Hybrid-retrieval outputs must live directly under artifacts.root")
    return HybridRetrievalConfig(seed, source, tfidf, fusion, artifacts, path)
