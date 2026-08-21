"""Strict validation-only configuration for Phase 7 candidate retrieval."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from shopee_match.errors import ConfigurationError
from shopee_match.hashing import matches_frozen_sha256, sha256_file
from shopee_match.training.hard_negative_config import (
    HardNegativeExperimentConfig,
    load_hard_negative_experiment_config,
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


@dataclass(frozen=True, slots=True)
class CandidateSourceConfig:
    phase6_config_path: Path
    phase6_config_sha256: str
    checkpoint_path: Path
    checkpoint_sha256: str
    metrics_path: Path
    metrics_sha256: str
    mined_manifest_path: Path
    mined_manifest_sha256: str
    experiment: HardNegativeExperimentConfig
    metrics: dict[str, Any]


@dataclass(frozen=True, slots=True)
class EmbeddingConfig:
    device: str
    batch_size: int
    num_workers: int


@dataclass(frozen=True, slots=True)
class ExactIndexConfig:
    block_size: int


@dataclass(frozen=True, slots=True)
class FaissIndexConfig:
    m: int
    ef_construction: int
    ef_search_values: tuple[int, ...]
    threads: int
    rerank_buffer: int


@dataclass(frozen=True, slots=True)
class CandidateSelectionConfig:
    k_values: tuple[int, ...]
    target_recall: float
    maximum_approximate_recall_drop: float
    minimum_exact_candidate_agreement: float
    latency_query_count: int
    latency_repetitions: int


@dataclass(frozen=True, slots=True)
class CandidateArtifactConfig:
    root: Path
    embedding_cache: Path
    exact_index: Path
    faiss_index: Path
    faiss_metadata: Path
    metrics: Path
    review: Path
    report: Path


@dataclass(frozen=True, slots=True)
class CandidateRetrievalConfig:
    seed: int
    source: CandidateSourceConfig
    embedding: EmbeddingConfig
    exact: ExactIndexConfig
    faiss: FaissIndexConfig
    selection: CandidateSelectionConfig
    artifacts: CandidateArtifactConfig
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
    path = _relative_path(raw[name], f"source.{name}")
    expected = _digest(raw[f"{name}_sha256"], f"source.{name}_sha256")
    try:
        if portable_text:
            matches, actual = matches_frozen_sha256(path, expected)
        else:
            actual = sha256_file(path)
            matches = actual == expected
    except OSError as exc:
        raise ConfigurationError(f"Cannot read frozen Phase 7 source: {path}") from exc
    if not matches:
        raise ConfigurationError(
            f"Frozen Phase 7 source hash mismatch for {path}: expected {expected}, got {actual}"
        )
    return path, expected


def load_candidate_retrieval_config(path: Path) -> CandidateRetrievalConfig:
    """Load and verify the Phase 7 source, validation protocol, and output contract."""
    root = _read_yaml(path, "candidate retrieval config")
    _only_keys(
        root,
        {
            "config_version",
            "seed",
            "source",
            "data",
            "embedding",
            "exact",
            "faiss",
            "selection",
            "artifacts",
        },
        "config",
    )
    if root["config_version"] != "phase7.candidate_retrieval.v1":
        raise ConfigurationError("Unsupported candidate retrieval config_version")
    seed = _nonnegative_int(root["seed"], "seed")

    source_raw = _mapping(root["source"], "source")
    _only_keys(
        source_raw,
        {
            "phase6_config",
            "phase6_config_sha256",
            "checkpoint",
            "checkpoint_sha256",
            "metrics",
            "metrics_sha256",
            "mined_manifest",
            "mined_manifest_sha256",
        },
        "source",
    )
    phase6_path, phase6_sha = _verified_file(source_raw, "phase6_config", portable_text=True)
    checkpoint_path, checkpoint_sha = _verified_file(source_raw, "checkpoint")
    metrics_path, metrics_sha = _verified_file(source_raw, "metrics")
    manifest_path, manifest_sha = _verified_file(source_raw, "mined_manifest")
    experiment = load_hard_negative_experiment_config(phase6_path)
    try:
        metrics = cast(dict[str, Any], json.loads(metrics_path.read_text(encoding="utf-8")))
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigurationError("Cannot load frozen Phase 6 metrics") from exc
    if (
        seed != experiment.seed
        or checkpoint_path != experiment.artifacts.checkpoint
        or metrics_path != experiment.artifacts.metrics
        or manifest_path != experiment.artifacts.manifest
        or metrics.get("pipeline_version") != "phase6.hard_negative_training.v1"
        or metrics.get("acceptance", {}).get("pilot_pass") is not True
        or metrics.get("test", {}).get("status") != "disabled_phase6_validation_only"
        or metrics.get("data", {}).get("test_accessed") is not False
    ):
        raise ConfigurationError("Phase 7 source is not the accepted validation-only Phase 6 run")
    source = CandidateSourceConfig(
        phase6_config_path=phase6_path,
        phase6_config_sha256=phase6_sha,
        checkpoint_path=checkpoint_path,
        checkpoint_sha256=checkpoint_sha,
        metrics_path=metrics_path,
        metrics_sha256=metrics_sha,
        mined_manifest_path=manifest_path,
        mined_manifest_sha256=manifest_sha,
        experiment=experiment,
        metrics=metrics,
    )

    data_raw = _mapping(root["data"], "data")
    _only_keys(data_raw, {"split", "evaluate_test"}, "data")
    if data_raw["split"] != "validation" or data_raw["evaluate_test"] is not False:
        raise ConfigurationError("Phase 7 model/index selection may use validation only")

    embedding_raw = _mapping(root["embedding"], "embedding")
    _only_keys(embedding_raw, {"device", "batch_size", "num_workers"}, "embedding")
    device = _typed(embedding_raw["device"], str, "embedding.device")
    if device not in {"auto", "cpu", "cuda"}:
        raise ConfigurationError("embedding.device must be auto, cpu, or cuda")
    embedding = EmbeddingConfig(
        device=device,
        batch_size=_positive_int(embedding_raw["batch_size"], "embedding.batch_size"),
        num_workers=_nonnegative_int(embedding_raw["num_workers"], "embedding.num_workers"),
    )

    exact_raw = _mapping(root["exact"], "exact")
    _only_keys(exact_raw, {"block_size"}, "exact")
    exact = ExactIndexConfig(block_size=_positive_int(exact_raw["block_size"], "exact.block_size"))

    faiss_raw = _mapping(root["faiss"], "faiss")
    _only_keys(
        faiss_raw,
        {
            "index_type",
            "metric",
            "m",
            "ef_construction",
            "ef_search_values",
            "threads",
            "rerank_buffer",
        },
        "faiss",
    )
    if faiss_raw["index_type"] != "hnsw_flat" or faiss_raw["metric"] != "inner_product":
        raise ConfigurationError("Phase 7 supports normalized HNSW inner-product search only")
    ef_raw = _typed(faiss_raw["ef_search_values"], list, "faiss.ef_search_values")
    ef_values = tuple(
        _positive_int(value, f"faiss.ef_search_values[{index}]")
        for index, value in enumerate(ef_raw)
    )
    if tuple(sorted(set(ef_values))) != ef_values:
        raise ConfigurationError("faiss.ef_search_values must be sorted and unique")
    faiss = FaissIndexConfig(
        m=_positive_int(faiss_raw["m"], "faiss.m"),
        ef_construction=_positive_int(faiss_raw["ef_construction"], "faiss.ef_construction"),
        ef_search_values=ef_values,
        threads=_positive_int(faiss_raw["threads"], "faiss.threads"),
        rerank_buffer=_nonnegative_int(faiss_raw["rerank_buffer"], "faiss.rerank_buffer"),
    )

    selection_raw = _mapping(root["selection"], "selection")
    _only_keys(
        selection_raw,
        {
            "k_values",
            "target_recall",
            "maximum_approximate_recall_drop",
            "minimum_exact_candidate_agreement",
            "latency_query_count",
            "latency_repetitions",
        },
        "selection",
    )
    k_raw = _typed(selection_raw["k_values"], list, "selection.k_values")
    k_values = tuple(
        _positive_int(value, f"selection.k_values[{index}]") for index, value in enumerate(k_raw)
    )
    if tuple(sorted(set(k_values))) != k_values:
        raise ConfigurationError("selection.k_values must be sorted and unique")
    selection = CandidateSelectionConfig(
        k_values=k_values,
        target_recall=_fraction(selection_raw["target_recall"], "selection.target_recall"),
        maximum_approximate_recall_drop=_fraction(
            selection_raw["maximum_approximate_recall_drop"],
            "selection.maximum_approximate_recall_drop",
        ),
        minimum_exact_candidate_agreement=_fraction(
            selection_raw["minimum_exact_candidate_agreement"],
            "selection.minimum_exact_candidate_agreement",
        ),
        latency_query_count=_positive_int(
            selection_raw["latency_query_count"], "selection.latency_query_count"
        ),
        latency_repetitions=_positive_int(
            selection_raw["latency_repetitions"], "selection.latency_repetitions"
        ),
    )

    artifact_raw = _mapping(root["artifacts"], "artifacts")
    _only_keys(
        artifact_raw,
        {
            "root",
            "embedding_cache",
            "exact_index",
            "faiss_index",
            "faiss_metadata",
            "metrics",
            "review",
            "report",
        },
        "artifacts",
    )
    artifacts = CandidateArtifactConfig(
        root=_relative_path(artifact_raw["root"], "artifacts.root"),
        embedding_cache=_relative_path(
            artifact_raw["embedding_cache"], "artifacts.embedding_cache"
        ),
        exact_index=_relative_path(artifact_raw["exact_index"], "artifacts.exact_index"),
        faiss_index=_relative_path(artifact_raw["faiss_index"], "artifacts.faiss_index"),
        faiss_metadata=_relative_path(artifact_raw["faiss_metadata"], "artifacts.faiss_metadata"),
        metrics=_relative_path(artifact_raw["metrics"], "artifacts.metrics"),
        review=_relative_path(artifact_raw["review"], "artifacts.review"),
        report=_relative_path(artifact_raw["report"], "artifacts.report"),
    )
    local_outputs = (
        artifacts.embedding_cache,
        artifacts.exact_index,
        artifacts.faiss_index,
        artifacts.faiss_metadata,
        artifacts.metrics,
        artifacts.review,
    )
    if any(output.parent != artifacts.root for output in local_outputs):
        raise ConfigurationError("Phase 7 local outputs must live directly under artifacts.root")
    return CandidateRetrievalConfig(
        seed=seed,
        source=source,
        embedding=embedding,
        exact=exact,
        faiss=faiss,
        selection=selection,
        artifacts=artifacts,
        config_path=path,
    )
