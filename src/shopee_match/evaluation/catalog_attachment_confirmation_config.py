"""Frozen one-time confirmation contract for catalog attachment protocol v4."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from shopee_match.errors import ConfigurationError
from shopee_match.evaluation.catalog_attachment_config import (
    CatalogAttachmentConfig,
    CatalogAttachmentSafety,
    load_catalog_attachment_config,
)
from shopee_match.hashing import canonical_text_sha256, sha256_file
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
class ConfirmationPolicy:
    candidate_k: int
    metric_k_values: tuple[int, ...]
    pair_probability_threshold: float
    manual_review_margin: float
    target_margin: float
    variant_conflict_override_probability: float


@dataclass(frozen=True, slots=True)
class ConfirmationStability:
    maximum_attachment_precision_drop: float
    maximum_attachment_recall_drop: float
    maximum_attachment_f1_drop: float
    maximum_new_entity_detection_recall_drop: float
    maximum_new_entity_false_attachment_rate_increase: float
    maximum_overall_false_attachment_rate_increase: float
    maximum_manual_review_rate_increase: float


@dataclass(frozen=True, slots=True)
class ConfirmationRuntime:
    device: str
    extraction_batch_size: int
    pair_batch_size: int


@dataclass(frozen=True, slots=True)
class ConfirmationArtifacts:
    root: Path
    access_marker: Path
    metrics: Path
    report: Path


@dataclass(frozen=True, slots=True)
class CatalogAttachmentConfirmationConfig:
    seed: int
    attempt_number: int
    development_config_path: Path
    development_config: CatalogAttachmentConfig
    development_metrics_path: Path
    development_metrics: dict[str, Any]
    protocol_config_path: Path
    role_manifest: Path
    runtime: ConfirmationRuntime
    policy: ConfirmationPolicy
    safety: CatalogAttachmentSafety
    stability: ConfirmationStability
    artifacts: ConfirmationArtifacts
    config_path: Path


def _verified(raw: dict[str, Any], name: str, location: str, *, text: bool = False) -> Path:
    path = _relative_path(raw[name], f"{location}.{name}")
    expected = _typed(raw[f"{name}_sha256"], str, f"{location}.{name}_sha256").lower()
    actual = canonical_text_sha256(path) if text else sha256_file(path)
    if actual != expected:
        raise ConfigurationError(
            f"Confirmation source mismatch for {path}: expected {expected}, got {actual}"
        )
    return path


def _fraction(value: object, location: str) -> float:
    result = _number(value, location)
    if not 0.0 <= result <= 1.0:
        raise ConfigurationError(f"{location} must be inside [0, 1]")
    return result


def load_catalog_attachment_confirmation_config(
    path: Path,
) -> CatalogAttachmentConfirmationConfig:
    root = _read_yaml(path, "catalog-attachment confirmation config")
    _only_keys(
        root,
        {
            "config_version",
            "seed",
            "attempt_number",
            "source",
            "data",
            "runtime",
            "policy",
            "safety",
            "stability",
            "artifacts",
        },
        "config",
    )
    if root["config_version"] != "catalog_attachment.confirmation.v4":
        raise ConfigurationError("Unsupported catalog-attachment confirmation version")

    source = _mapping(root["source"], "source")
    names = {
        "development_config",
        "development_metrics",
        "protocol_config",
        "role_manifest",
    }
    _only_keys(source, names | {f"{name}_sha256" for name in names}, "source")
    development_config_path = _verified(source, "development_config", "source", text=True)
    development_metrics_path = _verified(source, "development_metrics", "source")
    protocol_config_path = _verified(source, "protocol_config", "source", text=True)
    role_manifest = _verified(source, "role_manifest", "source")
    development_config = load_catalog_attachment_config(development_config_path)
    development_metrics = cast(
        dict[str, Any], json.loads(development_metrics_path.read_text(encoding="utf-8"))
    )
    if (
        development_config.model_variant != "canonical"
        or development_metrics.get("pipeline_version") != "catalog_attachment.evaluation.v4"
        or development_metrics.get("model_variant") != "canonical"
        or development_metrics.get("status") != "development_policy_selected"
        or development_metrics.get("provenance", {}).get("config_sha256")
        != canonical_text_sha256(development_config_path)
    ):
        raise ConfigurationError("Confirmation requires accepted canonical development evidence")

    data = _mapping(root["data"], "data")
    _only_keys(
        data,
        {"source_split", "partition", "selection_enabled", "historical_test_access"},
        "data",
    )
    if data != {
        "source_split": "test",
        "partition": "confirmation",
        "selection_enabled": False,
        "historical_test_access": False,
    }:
        raise ConfigurationError("Confirmation data contract must disable selection and history")

    runtime_raw = _mapping(root["runtime"], "runtime")
    _only_keys(
        runtime_raw,
        {"device", "extraction_batch_size", "pair_batch_size"},
        "runtime",
    )
    device = _typed(runtime_raw["device"], str, "runtime.device")
    if device not in {"cpu", "cuda", "auto"}:
        raise ConfigurationError("runtime.device must be cpu, cuda, or auto")
    runtime = ConfirmationRuntime(
        device,
        _positive_int(runtime_raw["extraction_batch_size"], "runtime.extraction_batch_size"),
        _positive_int(runtime_raw["pair_batch_size"], "runtime.pair_batch_size"),
    )

    policy_raw = _mapping(root["policy"], "policy")
    policy_names = {
        "candidate_k",
        "metric_k_values",
        "pair_probability_threshold",
        "manual_review_margin",
        "target_margin",
        "variant_conflict_override_probability",
    }
    _only_keys(policy_raw, policy_names, "policy")
    metric_k = tuple(
        _positive_int(value, f"policy.metric_k_values[{index}]")
        for index, value in enumerate(cast(list[object], policy_raw["metric_k_values"]))
    )
    policy = ConfirmationPolicy(
        _positive_int(policy_raw["candidate_k"], "policy.candidate_k"),
        metric_k,
        _fraction(policy_raw["pair_probability_threshold"], "policy.pair_probability_threshold"),
        _fraction(policy_raw["manual_review_margin"], "policy.manual_review_margin"),
        _fraction(policy_raw["target_margin"], "policy.target_margin"),
        _fraction(
            policy_raw["variant_conflict_override_probability"],
            "policy.variant_conflict_override_probability",
        ),
    )
    selected = development_metrics["selection"]["selected"]
    if (
        policy.candidate_k != development_config.policy.candidate_k
        or policy.metric_k_values != development_config.policy.metric_k_values
        or policy.pair_probability_threshold != selected["threshold"]
        or policy.manual_review_margin != development_config.policy.manual_review_margin
        or policy.target_margin != development_config.policy.target_margin
        or policy.variant_conflict_override_probability
        != development_config.policy.variant_conflict_override_probability
    ):
        raise ConfigurationError("Confirmation policy differs from development selection")

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
            safety_raw["minimum_attachment_precision"], "safety.minimum_attachment_precision"
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
            safety_raw["maximum_manual_review_rate"], "safety.maximum_manual_review_rate"
        ),
    )
    if safety != development_config.safety:
        raise ConfigurationError("Confirmation absolute safety gates differ from development")

    stability_raw = _mapping(root["stability"], "stability")
    stability_names = {
        "maximum_attachment_precision_drop",
        "maximum_attachment_recall_drop",
        "maximum_attachment_f1_drop",
        "maximum_new_entity_detection_recall_drop",
        "maximum_new_entity_false_attachment_rate_increase",
        "maximum_overall_false_attachment_rate_increase",
        "maximum_manual_review_rate_increase",
    }
    _only_keys(stability_raw, stability_names, "stability")
    stability = ConfirmationStability(
        **{name: _fraction(stability_raw[name], f"stability.{name}") for name in stability_names}
    )

    artifacts_raw = _mapping(root["artifacts"], "artifacts")
    _only_keys(artifacts_raw, {"root", "access_marker", "metrics", "report"}, "artifacts")
    artifact_root = _relative_path(artifacts_raw["root"], "artifacts.root")
    access_marker = _relative_path(artifacts_raw["access_marker"], "artifacts.access_marker")
    metrics = _relative_path(artifacts_raw["metrics"], "artifacts.metrics")
    report = _relative_path(artifacts_raw["report"], "artifacts.report")
    if any(output.parent != artifact_root for output in (access_marker, metrics, report)):
        raise ConfigurationError("Confirmation outputs must live directly under artifacts.root")
    return CatalogAttachmentConfirmationConfig(
        seed=_positive_int(root["seed"], "seed"),
        attempt_number=_positive_int(root["attempt_number"], "attempt_number"),
        development_config_path=development_config_path,
        development_config=development_config,
        development_metrics_path=development_metrics_path,
        development_metrics=development_metrics,
        protocol_config_path=protocol_config_path,
        role_manifest=role_manifest,
        runtime=runtime,
        policy=policy,
        safety=safety,
        stability=stability,
        artifacts=ConfirmationArtifacts(artifact_root, access_marker, metrics, report),
        config_path=path,
    )
