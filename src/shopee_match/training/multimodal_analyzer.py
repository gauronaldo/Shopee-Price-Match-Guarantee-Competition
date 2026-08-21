"""Validation-only failure analysis for the selected Phase 5 checkpoint."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

import torch
from torch.utils.data import DataLoader

from shopee_match.errors import ConfigurationError
from shopee_match.evaluation.embedding_retrieval import rank_cosine_embeddings
from shopee_match.evaluation.multimodal_failure_analysis import (
    build_multimodal_failure_review,
)
from shopee_match.evaluation.multimodal_retrieval import (
    rank_simple_score_fusion,
    rerank_with_pair_head,
    unimodal_rankings,
)
from shopee_match.evaluation.protocol import load_splits, retrieval_metrics
from shopee_match.models import LearnedMultimodalFusion
from shopee_match.training.multimodal_config import load_multimodal_experiment_config
from shopee_match.training.multimodal_data import load_cached_multimodal_split
from shopee_match.training.multimodal_trainer import (
    _resolve_device,
    _write_text_atomic,
    extract_joint_embeddings,
)


def _render_report(review: dict[str, Any], metrics: dict[str, dict[str, float]]) -> str:
    counts = review["counts"]
    query_count = review["queries"]
    rows = "\n".join(
        f"| {name.replace('_', ' ').title()} | {count:,} | {count / query_count:.2%} |"
        for name, count in counts.items()
    )
    retrieval_rows = "\n".join(
        f"| {label} | {metrics[key]['map@20']:.5f} | {metrics[key]['recall@20']:.5f} |"
        for key, label in (
            ("image", "Image only"),
            ("text", "Text only"),
            ("simple_fusion", "Simple score fusion"),
            ("learned_fusion", "Learned fusion"),
            ("pair_head", "Pair-head rerank"),
        )
    )
    return f"""# Multimodal fusion validation failure analysis

## Scope

This analysis uses only the validation split and the selected Phase 5 checkpoint. Categories can
overlap because one query may exhibit both a modality disagreement and a variant-token conflict.
The title-rich review records are local artifacts and are not included in Git.

## Retrieval context

| Method | mAP@20 | Recall@20 |
|---|---:|---:|
{retrieval_rows}

## Categorized diagnostics

| Category | Queries | Share of validation queries |
|---|---:|---:|
{rows}

## Interpretation

- `pair top1 false match` is a ranking error at the first result, not necessarily a complete
  retrieval failure.
- `pair retrieval miss` means no true duplicate appears in the pair head's Top-20 candidate list.
- `pair head regression/rescue` measures whether learned reranking hurts or fixes the simple-fusion
  Top-1 result.
- `image rescue` and `text rescue` expose cases where one modality is correct and the other is not.
- `variant token conflict` is an automatic diagnostic: the false Top-1 pair differs in digits or
  units. It is evidence for manual review, not a semantic ground-truth category.

The key Phase 5 risk is the pair head improving average ranking while moving a subset of already
correct simple-fusion queries in the wrong direction. Phase 6 hard-negative mining should target
these regressions, especially variant conflicts involving model numbers, quantity, size, or unit.
"""


def run_multimodal_validation_failure_analysis(config_path: Path) -> dict[str, Any]:
    """Write aggregate report plus local title-rich records; never load test."""
    config = load_multimodal_experiment_config(config_path)
    checkpoint_path = config.artifacts.root / "best.pt"
    metrics_path = config.artifacts.root / "metrics.json"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("checkpoint_version") != "phase5.scratch_multimodal_checkpoint.v1":
        raise ConfigurationError("Unsupported multimodal checkpoint version")
    if checkpoint.get("model_spec") != asdict(config.model_spec):
        raise ConfigurationError("Checkpoint architecture differs from the analysis config")
    expected_sources = {
        "image_sha256": config.frozen.image_config.checkpoint.sha256,
        "text_sha256": config.frozen.text_config.checkpoint.sha256,
    }
    if checkpoint.get("source_checkpoints") != expected_sources:
        raise ConfigurationError("Checkpoint source encoders differ from the analysis config")

    splits = load_splits(config.data.metadata_csv, config.data.split_manifest)
    validation = splits["validation"]
    dataset = load_cached_multimodal_split(config, "validation")
    loader = DataLoader(dataset, batch_size=512, shuffle=False, num_workers=0)
    device = _resolve_device(config.training.device)
    model = LearnedMultimodalFusion(config.model_spec)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    posting_ids, image, text, joint, _seconds = extract_joint_embeddings(model, loader, device)
    expected_ids = tuple(item.posting_id for item in validation.items)
    if posting_ids != expected_ids:
        raise ConfigurationError("Cached validation order differs from the frozen split")

    recorded = cast(dict[str, Any], json.loads(metrics_path.read_text(encoding="utf-8")))
    image_weight = float(
        recorded["validation"]["base_ablations"]["simple_score_fusion"]["selected_image_weight"]
    )
    image_ranking, text_ranking = unimodal_rankings(
        posting_ids, image, text, config.evaluation.candidate_k
    )
    simple_ranking = rank_simple_score_fusion(
        posting_ids,
        image,
        text,
        image_weight=image_weight,
        candidate_k=config.evaluation.candidate_k,
    )
    learned_ranking = rank_cosine_embeddings(posting_ids, joint, config.evaluation.candidate_k)
    pair_ranking = rerank_with_pair_head(model, posting_ids, joint, learned_ranking, device)
    rankings = {
        "image": image_ranking,
        "text": text_ranking,
        "simple_fusion": simple_ranking,
        "learned_fusion": learned_ranking,
        "pair_head": pair_ranking,
    }
    review = build_multimodal_failure_review(rankings, validation)
    metrics = {
        name: retrieval_metrics(
            ranking,
            validation.label_by_id,
            config.evaluation.recall_at,
            config.evaluation.average_precision_at,
        )
        for name, ranking in rankings.items()
    }
    output_path = config.artifacts.root / "validation_failure_review.json"
    report_path = Path("reports/multimodal_fusion_failure_analysis.md")
    _write_text_atomic(
        output_path, json.dumps(review, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )
    _write_text_atomic(report_path, _render_report(review, metrics))
    return {
        "status": "complete",
        "split": "validation",
        "output": str(output_path),
        "report": str(report_path),
        "counts": review["counts"],
        "test_accessed": False,
    }
