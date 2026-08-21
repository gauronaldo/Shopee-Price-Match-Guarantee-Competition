"""Strict configuration for the frozen Phase 11 demonstration service."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from shopee_match.clustering.config import EntityResolutionConfig, load_entity_resolution_config
from shopee_match.errors import ConfigurationError
from shopee_match.hashing import matches_frozen_sha256, sha256_file
from shopee_match.training.text_config import (
    _mapping,
    _number,
    _only_keys,
    _positive_int,
    _read_yaml,
    _relative_path,
    _typed,
)


@dataclass(frozen=True, slots=True)
class DemoSourceConfig:
    entity_config_path: Path
    entity_config_sha256: str
    entity_metrics_path: Path
    entity_metrics_sha256: str
    modality_embeddings_path: Path
    modality_embeddings_sha256: str
    entity_assignments_path: Path
    entity_assignments_sha256: str
    experiment: EntityResolutionConfig
    metrics: dict[str, Any]


@dataclass(frozen=True, slots=True)
class DemoIndexConfig:
    backend: str
    m: int
    ef_construction: int
    ef_search: int
    threads: int
    rerank_buffer: int


@dataclass(frozen=True, slots=True)
class DemoPolicyConfig:
    candidate_k: int
    default_top_k: int
    maximum_top_k: int
    pair_probability_threshold: float
    reciprocal_rank: int
    variant_conflict_override_probability: float
    manual_review_margin: float


@dataclass(frozen=True, slots=True)
class DemoRuntimeConfig:
    device: str
    maximum_upload_bytes: int
    maximum_batch_size: int


@dataclass(frozen=True, slots=True)
class DemoConfig:
    source: DemoSourceConfig
    index: DemoIndexConfig
    policy: DemoPolicyConfig
    runtime: DemoRuntimeConfig
    config_path: Path


def _digest(value: Any, location: str) -> str:
    digest = _typed(value, str, location).lower()
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ConfigurationError(f"{location} must be a SHA-256 digest")
    return digest


def _fraction(value: Any, location: str) -> float:
    result = _number(value, location, allow_zero=True)
    if result > 1:
        raise ConfigurationError(f"{location} must be inside [0, 1]")
    return result


def _verified_path(
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
        raise ConfigurationError(f"Cannot read frozen demo source: {path}") from exc
    if not matches:
        raise ConfigurationError(
            f"Frozen demo source hash mismatch for {path}: expected {expected}, got {actual}"
        )
    return path, expected


def load_demo_config(path: Path) -> DemoConfig:
    """Load a validation-catalog demo without exposing ground-truth labels to inference."""
    root = _read_yaml(path, "demo config")
    _only_keys(
        root,
        {"config_version", "source", "catalog", "index", "policy", "runtime"},
        "config",
    )
    if root["config_version"] != "demo.inference.v1":
        raise ConfigurationError("Unsupported demo config_version")

    source_raw = _mapping(root["source"], "source")
    source_names = {
        "entity_config",
        "entity_metrics",
        "modality_embeddings",
        "entity_assignments",
    }
    _only_keys(
        source_raw,
        source_names | {f"{name}_sha256" for name in source_names},
        "source",
    )
    entity_path, entity_sha = _verified_path(source_raw, "entity_config", portable_text=True)
    metrics_path, metrics_sha = _verified_path(source_raw, "entity_metrics")
    modality_path, modality_sha = _verified_path(source_raw, "modality_embeddings")
    assignments_path, assignments_sha = _verified_path(source_raw, "entity_assignments")
    entity = load_entity_resolution_config(entity_path)
    if metrics_path != entity.artifacts.metrics or assignments_path != entity.artifacts.assignments:
        raise ConfigurationError("Demo metrics and assignments must come from the entity run")
    try:
        metrics = cast(dict[str, Any], json.loads(metrics_path.read_text(encoding="utf-8")))
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigurationError("Cannot parse frozen entity metrics") from exc
    if (
        metrics.get("pipeline_version") != "phase8.entity_resolution.v1"
        or metrics.get("status") != "phase8_complete_validation_only"
        or metrics.get("data", {}).get("split") != "validation"
        or metrics.get("data", {}).get("test_accessed") is not False
    ):
        raise ConfigurationError("Demo must use the accepted validation-only entity run")

    catalog_raw = _mapping(root["catalog"], "catalog")
    _only_keys(catalog_raw, {"split"}, "catalog")
    if catalog_raw["split"] != "validation":
        raise ConfigurationError("The showcase demo intentionally uses the validation catalog")

    index_raw = _mapping(root["index"], "index")
    _only_keys(
        index_raw,
        {"backend", "m", "ef_construction", "ef_search", "threads", "rerank_buffer"},
        "index",
    )
    backend = _typed(index_raw["backend"], str, "index.backend")
    if backend not in {"exact", "faiss_hnsw"}:
        raise ConfigurationError("index.backend must be exact or faiss_hnsw")
    index = DemoIndexConfig(
        backend=backend,
        m=_positive_int(index_raw["m"], "index.m"),
        ef_construction=_positive_int(index_raw["ef_construction"], "index.ef_construction"),
        ef_search=_positive_int(index_raw["ef_search"], "index.ef_search"),
        threads=_positive_int(index_raw["threads"], "index.threads"),
        rerank_buffer=_positive_int(index_raw["rerank_buffer"], "index.rerank_buffer"),
    )

    policy_raw = _mapping(root["policy"], "policy")
    _only_keys(
        policy_raw,
        {"candidate_k", "default_top_k", "maximum_top_k"},
        "policy",
    )
    selected = metrics["selection"]["selected"]
    candidate_k = _positive_int(policy_raw["candidate_k"], "policy.candidate_k")
    default_top_k = _positive_int(policy_raw["default_top_k"], "policy.default_top_k")
    maximum_top_k = _positive_int(policy_raw["maximum_top_k"], "policy.maximum_top_k")
    selected_candidate_k = int(entity.source.metrics["selection"]["candidate_k"])
    if candidate_k != selected_candidate_k or not default_top_k <= maximum_top_k <= candidate_k:
        raise ConfigurationError("Demo K values must preserve the frozen candidate K")
    policy = DemoPolicyConfig(
        candidate_k=candidate_k,
        default_top_k=default_top_k,
        maximum_top_k=maximum_top_k,
        pair_probability_threshold=_fraction(
            selected["pair_probability_threshold"], "selected.pair_probability_threshold"
        ),
        reciprocal_rank=_positive_int(selected["reciprocal_rank"], "selected.reciprocal_rank"),
        variant_conflict_override_probability=entity.selection.variant_conflict_override_probability,
        manual_review_margin=entity.selection.manual_review_margin,
    )

    runtime_raw = _mapping(root["runtime"], "runtime")
    _only_keys(runtime_raw, {"device", "maximum_upload_mb", "maximum_batch_size"}, "runtime")
    device = _typed(runtime_raw["device"], str, "runtime.device")
    if device not in {"auto", "cpu", "cuda"}:
        raise ConfigurationError("runtime.device must be auto, cpu, or cuda")
    upload_mb = _positive_int(runtime_raw["maximum_upload_mb"], "runtime.maximum_upload_mb")
    runtime = DemoRuntimeConfig(
        device=device,
        maximum_upload_bytes=upload_mb * 1024 * 1024,
        maximum_batch_size=_positive_int(
            runtime_raw["maximum_batch_size"], "runtime.maximum_batch_size"
        ),
    )
    return DemoConfig(
        source=DemoSourceConfig(
            entity_config_path=entity_path,
            entity_config_sha256=entity_sha,
            entity_metrics_path=metrics_path,
            entity_metrics_sha256=metrics_sha,
            modality_embeddings_path=modality_path,
            modality_embeddings_sha256=modality_sha,
            entity_assignments_path=assignments_path,
            entity_assignments_sha256=assignments_sha,
            experiment=entity,
            metrics=metrics,
        ),
        index=index,
        policy=policy,
        runtime=runtime,
        config_path=path,
    )
