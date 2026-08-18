"""Strict configuration for deterministic classical retrieval benchmarks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, TypeVar

import yaml

from shopee_match.errors import ConfigurationError

T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class EvaluationConfig:
    tune_split: str
    final_split: str
    recall_at: tuple[int, ...]
    average_precision_at: int


@dataclass(frozen=True, slots=True)
class PhashConfig:
    candidate_k: int


@dataclass(frozen=True, slots=True)
class TfidfConfig:
    ngram_range: tuple[int, int]
    max_features: int
    candidate_k: int


@dataclass(frozen=True, slots=True)
class OrbConfig:
    features: int
    candidate_k_per_source: int
    top_k: int


@dataclass(frozen=True, slots=True)
class FusionConfig:
    weight_grid: tuple[float, ...]
    top_k: int


@dataclass(frozen=True, slots=True)
class ArtifactConfig:
    root: Path
    report: Path
    threshold_figure: Path


@dataclass(frozen=True, slots=True)
class ClassicalRetrievalConfig:
    config_version: str
    seed: int
    metadata_csv: Path
    split_manifest: Path
    image_dir: Path
    evaluation: EvaluationConfig
    phash: PhashConfig
    tfidf: TfidfConfig
    orb: OrbConfig
    fusion: FusionConfig
    artifacts: ArtifactConfig
    config_path: Path


def _mapping(value: Any, location: str) -> dict[str, Any]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ConfigurationError(f"{location} must be a mapping with string keys")
    return value


def _only_keys(value: dict[str, Any], expected: set[str], location: str) -> None:
    missing, unknown = expected - set(value), set(value) - expected
    if missing:
        raise ConfigurationError(f"Missing keys in {location}: {sorted(missing)}")
    if unknown:
        raise ConfigurationError(f"Unknown keys in {location}: {sorted(unknown)}")


def _typed(value: Any, expected: type[T], location: str) -> T:
    if not isinstance(value, expected):
        raise ConfigurationError(f"{location} must be {expected.__name__}")
    return value


def _positive_int(value: Any, location: str) -> int:
    result = _typed(value, int, location)
    if isinstance(result, bool) or result <= 0:
        raise ConfigurationError(f"{location} must be a positive integer")
    return result


def _relative_path(value: Any, location: str) -> Path:
    raw = _typed(value, str, location)
    variants = (PurePosixPath(raw), PureWindowsPath(raw))
    if any(path.is_absolute() or ".." in path.parts for path in variants):
        raise ConfigurationError(f"{location} must be a project-relative path without '..'")
    return Path(raw)


def _weight(value: Any, location: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ConfigurationError(f"{location} must be numeric")
    result = float(value)
    if not 0 <= result <= 1:
        raise ConfigurationError(f"{location} must be in [0, 1]")
    return result


def load_classical_retrieval_config(path: Path) -> ClassicalRetrievalConfig:
    """Load a strict retrieval benchmark config without machine-specific paths."""
    try:
        raw: Any = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigurationError(f"Cannot load configuration {path}: {exc}") from exc
    root = _mapping(raw, "config")
    _only_keys(
        root,
        {"config_version", "seed", "data", "evaluation", "baselines", "artifacts"},
        "config",
    )

    seed = _typed(root["seed"], int, "seed")
    if isinstance(seed, bool) or not 0 <= seed <= 2**32 - 1:
        raise ConfigurationError("seed must be in [0, 2**32 - 1]")

    data = _mapping(root["data"], "data")
    _only_keys(data, {"metadata_csv", "split_manifest", "image_dir"}, "data")

    evaluation = _mapping(root["evaluation"], "evaluation")
    _only_keys(
        evaluation,
        {
            "tune_split",
            "final_split",
            "candidate_pool",
            "exclude_query_itself",
            "recall_at",
            "average_precision_at",
        },
        "evaluation",
    )
    tune_split = _typed(evaluation["tune_split"], str, "evaluation.tune_split")
    final_split = _typed(evaluation["final_split"], str, "evaluation.final_split")
    if (tune_split, final_split) != ("validation", "test"):
        raise ConfigurationError(
            "The classical benchmark must tune on validation and evaluate finally on test"
        )
    if (
        evaluation["candidate_pool"] != "full_split"
        or evaluation["exclude_query_itself"] is not True
    ):
        raise ConfigurationError(
            "The classical benchmark requires candidate_pool=full_split "
            "and exclude_query_itself=true"
        )
    recall_raw = _typed(evaluation["recall_at"], list, "evaluation.recall_at")
    recall_at = tuple(
        _positive_int(value, f"evaluation.recall_at[{index}]")
        for index, value in enumerate(recall_raw)
    )
    if not recall_at or tuple(sorted(set(recall_at))) != recall_at:
        raise ConfigurationError("evaluation.recall_at must be non-empty, sorted, and unique")
    average_precision_at = _positive_int(
        evaluation["average_precision_at"], "evaluation.average_precision_at"
    )

    baselines = _mapping(root["baselines"], "baselines")
    _only_keys(baselines, {"phash", "tfidf", "orb", "fusion"}, "baselines")
    phash = _mapping(baselines["phash"], "baselines.phash")
    _only_keys(phash, {"candidate_k"}, "baselines.phash")
    tfidf = _mapping(baselines["tfidf"], "baselines.tfidf")
    _only_keys(
        tfidf,
        {"analyzer", "ngram_range", "max_features", "candidate_k"},
        "baselines.tfidf",
    )
    if tfidf["analyzer"] != "char_wb":
        raise ConfigurationError("baselines.tfidf.analyzer must remain char_wb")
    ngram_raw = _typed(tfidf["ngram_range"], list, "baselines.tfidf.ngram_range")
    if len(ngram_raw) != 2:
        raise ConfigurationError("baselines.tfidf.ngram_range must contain two integers")
    ngram_range = tuple(
        _positive_int(value, f"baselines.tfidf.ngram_range[{index}]")
        for index, value in enumerate(ngram_raw)
    )
    if ngram_range[0] > ngram_range[1]:
        raise ConfigurationError("baselines.tfidf.ngram_range must be ascending")

    orb = _mapping(baselines["orb"], "baselines.orb")
    _only_keys(
        orb,
        {"features", "candidate_source", "candidate_k_per_source", "top_k"},
        "baselines.orb",
    )
    if orb["candidate_source"] != "union_phash_tfidf":
        raise ConfigurationError("baselines.orb.candidate_source must be union_phash_tfidf")

    fusion = _mapping(baselines["fusion"], "baselines.fusion")
    _only_keys(fusion, {"weight_grid", "top_k"}, "baselines.fusion")
    weight_raw = _typed(fusion["weight_grid"], list, "baselines.fusion.weight_grid")
    weights = tuple(
        _weight(value, f"baselines.fusion.weight_grid[{index}]")
        for index, value in enumerate(weight_raw)
    )
    if not weights or tuple(sorted(set(weights))) != weights:
        raise ConfigurationError(
            "baselines.fusion.weight_grid must be non-empty, sorted, and unique"
        )

    artifacts = _mapping(root["artifacts"], "artifacts")
    _only_keys(artifacts, {"root", "report", "threshold_figure"}, "artifacts")

    largest_metric_k = max(*recall_at, average_precision_at)
    candidate_ks = (
        _positive_int(phash["candidate_k"], "baselines.phash.candidate_k"),
        _positive_int(tfidf["candidate_k"], "baselines.tfidf.candidate_k"),
    )
    orb_top_k = _positive_int(orb["top_k"], "baselines.orb.top_k")
    fusion_top_k = _positive_int(fusion["top_k"], "baselines.fusion.top_k")
    if min(*candidate_ks, orb_top_k, fusion_top_k) < largest_metric_k:
        raise ConfigurationError("Every emitted ranking must cover the largest configured metric K")

    return ClassicalRetrievalConfig(
        config_version=_typed(root["config_version"], str, "config_version"),
        seed=seed,
        metadata_csv=_relative_path(data["metadata_csv"], "data.metadata_csv"),
        split_manifest=_relative_path(data["split_manifest"], "data.split_manifest"),
        image_dir=_relative_path(data["image_dir"], "data.image_dir"),
        evaluation=EvaluationConfig(
            tune_split,
            final_split,
            recall_at,
            average_precision_at,
        ),
        phash=PhashConfig(candidate_ks[0]),
        tfidf=TfidfConfig(
            (ngram_range[0], ngram_range[1]),
            _positive_int(tfidf["max_features"], "baselines.tfidf.max_features"),
            candidate_ks[1],
        ),
        orb=OrbConfig(
            _positive_int(orb["features"], "baselines.orb.features"),
            _positive_int(orb["candidate_k_per_source"], "baselines.orb.candidate_k_per_source"),
            orb_top_k,
        ),
        fusion=FusionConfig(weights, fusion_top_k),
        artifacts=ArtifactConfig(
            _relative_path(artifacts["root"], "artifacts.root"),
            _relative_path(artifacts["report"], "artifacts.report"),
            _relative_path(artifacts["threshold_figure"], "artifacts.threshold_figure"),
        ),
        config_path=path,
    )
