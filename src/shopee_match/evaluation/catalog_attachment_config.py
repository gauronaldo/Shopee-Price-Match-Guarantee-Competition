"""Strict configuration for development-only catalog-attachment evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from shopee_match.errors import ConfigurationError
from shopee_match.hashing import canonical_text_sha256, sha256_file
from shopee_match.retrieval.hybrid_config import HybridRetrievalConfig, load_hybrid_retrieval_config
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
class CatalogAttachmentRuntime:
    device: str
    extraction_batch_size: int
    pair_batch_size: int


@dataclass(frozen=True, slots=True)
class CatalogAttachmentPolicy:
    candidate_k: int
    metric_k_values: tuple[int, ...]
    thresholds: tuple[float, ...]
    manual_review_margin: float
    target_margin: float
    variant_conflict_override_probability: float


@dataclass(frozen=True, slots=True)
class CatalogAttachmentSafety:
    minimum_attachment_precision: float
    minimum_new_entity_detection_recall: float
    maximum_new_entity_false_attachment_rate: float
    maximum_overall_false_attachment_rate: float
    maximum_manual_review_rate: float


@dataclass(frozen=True, slots=True)
class CatalogAttachmentArtifacts:
    root: Path
    metrics: Path
    report: Path


@dataclass(frozen=True, slots=True)
class CatalogAttachmentConfig:
    seed: int
    metadata_csv: Path
    source_manifest: Path
    role_manifest: Path
    hybrid_config_path: Path
    hybrid: HybridRetrievalConfig
    model_variant: str
    full_joint_config: Path | None
    full_joint_checkpoint: Path | None
    runtime: CatalogAttachmentRuntime
    policy: CatalogAttachmentPolicy
    safety: CatalogAttachmentSafety
    artifacts: CatalogAttachmentArtifacts
    config_path: Path


def _verified(raw: dict[str, Any], name: str, *, text: bool = False) -> Path:
    path = _relative_path(raw[name], f"source.{name}")
    expected = _typed(raw[f"{name}_sha256"], str, f"source.{name}_sha256").lower()
    actual = canonical_text_sha256(path) if text else sha256_file(path)
    if actual != expected:
        raise ConfigurationError(
            f"Catalog-attachment source mismatch for {path}: expected {expected}, got {actual}"
        )
    return path


def _fraction(value: object, location: str) -> float:
    result = _number(value, location)
    if not 0.0 <= result <= 1.0:
        raise ConfigurationError(f"{location} must be inside [0, 1]")
    return result


def load_catalog_attachment_config(path: Path) -> CatalogAttachmentConfig:
    root = _read_yaml(path, "catalog-attachment evaluation config")
    _only_keys(
        root,
        {
            "config_version",
            "seed",
            "source",
            "data",
            "runtime",
            "retrieval",
            "decision",
            "safety",
            "artifacts",
        },
        "config",
    )
    if root["config_version"] != "catalog_attachment.evaluation.v4":
        raise ConfigurationError("Unsupported catalog-attachment evaluation version")
    source = _mapping(root["source"], "source")
    required = {
        "metadata_csv",
        "metadata_csv_sha256",
        "source_manifest",
        "source_manifest_sha256",
        "role_manifest",
        "role_manifest_sha256",
        "hybrid_config",
        "hybrid_config_sha256",
        "model_variant",
    }
    optional = {
        "full_joint_config",
        "full_joint_config_sha256",
        "full_joint_checkpoint",
        "full_joint_checkpoint_sha256",
    }
    if missing := required - set(source):
        raise ConfigurationError(f"Missing catalog-attachment sources: {sorted(missing)}")
    if unknown := set(source) - required - optional:
        raise ConfigurationError(f"Unknown catalog-attachment sources: {sorted(unknown)}")
    metadata = _verified(source, "metadata_csv")
    source_manifest = _verified(source, "source_manifest")
    role_manifest = _verified(source, "role_manifest")
    hybrid_path = _verified(source, "hybrid_config", text=True)
    hybrid = load_hybrid_retrieval_config(hybrid_path)
    variant = _typed(source["model_variant"], str, "source.model_variant")
    if variant not in {"canonical", "full_joint_candidate"}:
        raise ConfigurationError("model_variant must be canonical or full_joint_candidate")
    full_joint_config = None
    full_joint_checkpoint = None
    if variant == "full_joint_candidate":
        if not optional <= set(source):
            raise ConfigurationError("Full-joint candidate requires its config and checkpoint")
        full_joint_config = _verified(source, "full_joint_config", text=True)
        full_joint_checkpoint = _verified(source, "full_joint_checkpoint")
    elif set(source) & optional:
        raise ConfigurationError("Canonical evaluation must not reference a candidate checkpoint")

    data = _mapping(root["data"], "data")
    _only_keys(data, {"source_split", "partition", "evaluate_confirmation"}, "data")
    if data != {
        "source_split": "validation",
        "partition": "development",
        "evaluate_confirmation": False,
    }:
        raise ConfigurationError("Catalog-attachment development must not access confirmation")

    runtime_raw = _mapping(root["runtime"], "runtime")
    _only_keys(
        runtime_raw,
        {"device", "extraction_batch_size", "pair_batch_size"},
        "runtime",
    )
    device = _typed(runtime_raw["device"], str, "runtime.device")
    if device not in {"cpu", "cuda", "auto"}:
        raise ConfigurationError("runtime.device must be cpu, cuda, or auto")
    runtime = CatalogAttachmentRuntime(
        device,
        _positive_int(runtime_raw["extraction_batch_size"], "runtime.extraction_batch_size"),
        _positive_int(runtime_raw["pair_batch_size"], "runtime.pair_batch_size"),
    )

    retrieval = _mapping(root["retrieval"], "retrieval")
    _only_keys(retrieval, {"candidate_k", "metric_k_values"}, "retrieval")
    candidate_k = _positive_int(retrieval["candidate_k"], "retrieval.candidate_k")
    metric_k = tuple(
        _positive_int(value, f"retrieval.metric_k_values[{index}]")
        for index, value in enumerate(cast(list[object], retrieval["metric_k_values"]))
    )
    if not metric_k or tuple(sorted(set(metric_k))) != metric_k or max(metric_k) > candidate_k:
        raise ConfigurationError("metric_k_values must be sorted, unique, and no larger than K")

    decision = _mapping(root["decision"], "decision")
    _only_keys(
        decision,
        {
            "pair_probability_thresholds",
            "manual_review_margin",
            "target_margin",
            "variant_conflict_override_probability",
        },
        "decision",
    )
    thresholds = tuple(
        _fraction(value, f"decision.pair_probability_thresholds[{index}]")
        for index, value in enumerate(cast(list[object], decision["pair_probability_thresholds"]))
    )
    if not thresholds or tuple(sorted(set(thresholds))) != thresholds:
        raise ConfigurationError("Decision thresholds must be sorted and unique")
    policy = CatalogAttachmentPolicy(
        candidate_k,
        metric_k,
        thresholds,
        _fraction(decision["manual_review_margin"], "decision.manual_review_margin"),
        _fraction(decision["target_margin"], "decision.target_margin"),
        _fraction(
            decision["variant_conflict_override_probability"],
            "decision.variant_conflict_override_probability",
        ),
    )

    safety_raw = _mapping(root["safety"], "safety")
    safety_names = {
        "minimum_attachment_precision",
        "minimum_new_entity_detection_recall",
        "maximum_new_entity_false_attachment_rate",
        "maximum_overall_false_attachment_rate",
        "maximum_manual_review_rate",
    }
    _only_keys(safety_raw, safety_names, "safety")
    safety = CatalogAttachmentSafety(
        minimum_attachment_precision=_fraction(
            safety_raw["minimum_attachment_precision"],
            "safety.minimum_attachment_precision",
        ),
        minimum_new_entity_detection_recall=_fraction(
            safety_raw["minimum_new_entity_detection_recall"],
            "safety.minimum_new_entity_detection_recall",
        ),
        maximum_new_entity_false_attachment_rate=_fraction(
            safety_raw["maximum_new_entity_false_attachment_rate"],
            "safety.maximum_new_entity_false_attachment_rate",
        ),
        maximum_overall_false_attachment_rate=_fraction(
            safety_raw["maximum_overall_false_attachment_rate"],
            "safety.maximum_overall_false_attachment_rate",
        ),
        maximum_manual_review_rate=_fraction(
            safety_raw["maximum_manual_review_rate"],
            "safety.maximum_manual_review_rate",
        ),
    )

    artifacts_raw = _mapping(root["artifacts"], "artifacts")
    _only_keys(artifacts_raw, {"root", "metrics", "report"}, "artifacts")
    artifact_root = _relative_path(artifacts_raw["root"], "artifacts.root")
    metrics = _relative_path(artifacts_raw["metrics"], "artifacts.metrics")
    report = _relative_path(artifacts_raw["report"], "artifacts.report")
    if metrics.parent != artifact_root or report.parent != artifact_root:
        raise ConfigurationError("Catalog-attachment artifacts must live directly under root")
    return CatalogAttachmentConfig(
        seed=_positive_int(root["seed"], "seed"),
        metadata_csv=metadata,
        source_manifest=source_manifest,
        role_manifest=role_manifest,
        hybrid_config_path=hybrid_path,
        hybrid=hybrid,
        model_variant=variant,
        full_joint_config=full_joint_config,
        full_joint_checkpoint=full_joint_checkpoint,
        runtime=runtime,
        policy=policy,
        safety=safety,
        artifacts=CatalogAttachmentArtifacts(artifact_root, metrics, report),
        config_path=path,
    )
