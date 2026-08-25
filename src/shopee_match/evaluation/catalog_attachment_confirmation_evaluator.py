"""One-time frozen confirmation evaluation for catalog attachment protocol v4."""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from shopee_match.errors import DataValidationError, OutputConflictError
from shopee_match.evaluation.catalog_attachment_confirmation_config import (
    CatalogAttachmentConfirmationConfig,
    ConfirmationStability,
    load_catalog_attachment_confirmation_config,
)
from shopee_match.evaluation.catalog_attachment_evaluator import (
    _extract_embeddings,
    _hybrid_ranking,
    _load_models,
    _score_pairs,
    _write_atomic,
    attachment_metrics,
    passes_safety,
    retrieval_metrics,
    validate_ranking_contract,
)
from shopee_match.evaluation.catalog_attachment_protocol import (
    load_catalog_roles,
    validate_catalog_roles,
)
from shopee_match.evaluation.protocol import load_named_split
from shopee_match.hashing import canonical_text_sha256, sha256_file
from shopee_match.reproducibility import seed_everything
from shopee_match.training.multimodal_trainer import _git_state, _resolve_device


def _existing(config: CatalogAttachmentConfirmationConfig) -> list[str]:
    return [
        str(path)
        for path in (
            config.artifacts.access_marker,
            config.artifacts.metrics,
            config.artifacts.report,
        )
        if path.exists()
    ]


def preflight_catalog_attachment_confirmation(config_path: Path) -> dict[str, object]:
    config = load_catalog_attachment_confirmation_config(config_path)
    commit, dirty = _git_state()
    outputs = _existing(config)
    device = _resolve_device(config.runtime.device)
    return {
        "status": "ready" if not dirty and not outputs else "blocked",
        "git_commit": commit,
        "git_dirty": dirty,
        "device": str(device),
        "existing_outputs": outputs,
        "confirmation_rows_loaded": False,
        "confirmation_metrics_computed": False,
        "candidate_k": config.policy.candidate_k,
        "pair_probability_threshold": config.policy.pair_probability_threshold,
        "attempt_number": config.attempt_number,
    }


def passes_stability(
    confirmation: dict[str, float],
    development: dict[str, float],
    stability: ConfirmationStability,
) -> bool:
    return (
        development["attachment_precision"] - confirmation["attachment_precision"]
        <= stability.maximum_attachment_precision_drop
        and development["attachment_recall"] - confirmation["attachment_recall"]
        <= stability.maximum_attachment_recall_drop
        and development["attachment_f1"] - confirmation["attachment_f1"]
        <= stability.maximum_attachment_f1_drop
        and development["new_entity_detection_recall"] - confirmation["new_entity_detection_recall"]
        <= stability.maximum_new_entity_detection_recall_drop
        and confirmation["new_entity_false_attachment_rate"]
        - development["new_entity_false_attachment_rate"]
        <= stability.maximum_new_entity_false_attachment_rate_increase
        and confirmation["overall_false_attachment_rate"]
        - development["overall_false_attachment_rate"]
        <= stability.maximum_overall_false_attachment_rate_increase
        and confirmation["manual_review_rate"] - development["manual_review_rate"]
        <= stability.maximum_manual_review_rate_increase
    )


def _deltas(confirmation: dict[str, float], development: dict[str, float]) -> dict[str, float]:
    names = (
        "attachment_precision",
        "attachment_recall",
        "attachment_f1",
        "new_entity_detection_recall",
        "new_entity_false_attachment_rate",
        "overall_false_attachment_rate",
        "manual_review_rate",
    )
    return {name: confirmation[name] - development[name] for name in names}


def _report(run: dict[str, Any]) -> str:
    development = run["development_reference"]
    confirmation = run["confirmation"]["attachment"]
    rows = "\n".join(
        f"| {label} | {development[key]:.5f} | {confirmation[key]:.5f} | "
        f"{confirmation[key] - development[key]:+.5f} |"
        for label, key in (
            ("Attachment precision", "attachment_precision"),
            ("Attachment recall", "attachment_recall"),
            ("Attachment F1", "attachment_f1"),
            ("New-entity detection recall", "new_entity_detection_recall"),
            ("New-entity false-attachment rate", "new_entity_false_attachment_rate"),
            ("Overall false-attachment rate", "overall_false_attachment_rate"),
            ("Manual-review rate", "manual_review_rate"),
        )
    )
    return f"""# Catalog Attachment Confirmation

Status: **{run["status"]}**. The development-selected policy was applied without confirmation-time
threshold, candidate-budget, or model selection.

| Metric | Development | Confirmation | Delta |
|---|---:|---:|---:|
{rows}

Absolute safety gate: **{run["acceptance"]["passes_absolute_safety"]}**. Stability gate:
**{run["acceptance"]["passes_stability"]}**. Historical test data was not accessed.
"""


def run_catalog_attachment_confirmation(config_path: Path) -> dict[str, object]:
    config = load_catalog_attachment_confirmation_config(config_path)
    existing = _existing(config)
    if existing:
        raise OutputConflictError(
            "Confirmation access or output already exists; refusing to rerun: "
            + ", ".join(existing)
        )
    commit, dirty = _git_state()
    if dirty:
        raise DataValidationError("Catalog-attachment confirmation requires a clean Git worktree")
    config_sha = canonical_text_sha256(config.config_path)
    _write_atomic(
        config.artifacts.access_marker,
        json.dumps(
            {
                "status": "catalog_attachment_confirmation_access_started",
                "git_commit": commit,
                "git_dirty": False,
                "config_sha256": config_sha,
                "attempt_number": config.attempt_number,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
    )

    started = time.perf_counter()
    seed_everything(config.seed, deterministic=True)
    device = _resolve_device(config.runtime.device)
    development_config = config.development_config
    train = load_named_split(
        development_config.metadata_csv,
        development_config.source_manifest,
        "train",
    )
    confirmation = load_named_split(
        development_config.metadata_csv,
        development_config.source_manifest,
        "test",
    )
    roles = load_catalog_roles(config.role_manifest, partition="confirmation")
    validate_catalog_roles(confirmation, roles)
    (
        image_model,
        text_model,
        fusion_model,
        vocabulary,
        maximum_length,
        image_dir,
        image_size,
    ) = _load_models(development_config, device)
    posting_ids, embeddings = _extract_embeddings(
        confirmation,
        image_dir=image_dir,
        image_size=image_size,
        image_model=image_model,
        text_model=text_model,
        fusion_model=fusion_model,
        vocabulary=vocabulary,
        maximum_length=maximum_length,
        device=device,
        batch_size=config.runtime.extraction_batch_size,
    )
    ranking, references, queries = _hybrid_ranking(
        development_config,
        train,
        confirmation,
        posting_ids,
        embeddings,
        roles,
    )
    ranking_audit = validate_ranking_contract(
        ranking, queries, references, config.policy.candidate_k
    )
    retrieval = retrieval_metrics(
        ranking,
        queries,
        references,
        confirmation.label_by_id,
        config.policy.metric_k_values,
    )
    evidence = _score_pairs(
        fusion_model,
        posting_ids,
        embeddings,
        ranking,
        {item.posting_id: item for item in confirmation.items},
        device,
        config.runtime.pair_batch_size,
    )
    metrics = attachment_metrics(
        evidence,
        roles,
        confirmation.label_by_id,
        threshold=config.policy.pair_probability_threshold,
        manual_review_margin=config.policy.manual_review_margin,
        target_margin=config.policy.target_margin,
        variant_conflict_override_probability=config.policy.variant_conflict_override_probability,
    )
    development = config.development_metrics["selection"]["selected"]
    absolute = passes_safety(metrics, config.safety)
    stable = passes_stability(metrics, development, config.stability)
    status = "confirmation_passed" if absolute and stable else "confirmation_failed"
    run: dict[str, Any] = {
        "pipeline_version": "catalog_attachment.confirmation.v4",
        "status": status,
        "provenance": {
            "git_commit": commit,
            "git_dirty": False,
            "config_sha256": config_sha,
            "role_manifest_sha256": sha256_file(config.role_manifest),
        },
        "data": {
            "train_listings_for_statistics": len(train.items),
            "confirmation_listings": len(confirmation.items),
            "catalog_references": len(references),
            "queries": len(queries),
            "confirmation_accessed": True,
            "confirmation_metrics_computed": True,
            "historical_test_accessed": False,
            "selection_enabled": False,
            "label_group_used_by_retrieval": False,
            "train_confirmation_label_overlap": len(
                set(train.label_by_id.values()) & set(confirmation.label_by_id.values())
            ),
        },
        "frozen_policy": asdict(config.policy),
        "absolute_safety": asdict(config.safety),
        "stability_gates": asdict(config.stability),
        "development_reference": development,
        "confirmation": {
            "retrieval": retrieval,
            "attachment": metrics,
            "ranking_contract": ranking_audit,
        },
        "acceptance": {
            "passes_absolute_safety": absolute,
            "passes_stability": stable,
            "deltas": _deltas(metrics, development),
        },
        "runtime_seconds": time.perf_counter() - started,
    }
    _write_atomic(config.artifacts.metrics, json.dumps(run, indent=2, sort_keys=True) + "\n")
    _write_atomic(config.artifacts.report, _report(run))
    return {
        "status": status,
        "metrics": str(config.artifacts.metrics),
        "passes_absolute_safety": absolute,
        "passes_stability": stable,
        "confirmation_accessed": True,
    }
