"""Confirmatory held-out evaluation of the validation-frozen hybrid system."""

from __future__ import annotations

import json
import logging
import platform
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from shopee_match.clustering.benchmark import _failure_review
from shopee_match.clustering.graph import (
    build_conservative_clusters,
    score_candidate_pairs,
    scored_pair_payload,
)
from shopee_match.clustering.metrics import (
    candidate_pair_classification_metrics,
    clustering_metrics,
    edge_metrics,
    group_size_strata,
)
from shopee_match.errors import DataValidationError, OutputConflictError
from shopee_match.evaluation.final_system_evaluator import (
    _joint_embeddings,
    _write_assignments_atomic,
    _write_embeddings_atomic,
    _write_text_atomic,
)
from shopee_match.evaluation.hybrid_system_config import (
    HybridSystemArtifacts,
    load_hybrid_system_evaluation_config,
)
from shopee_match.evaluation.protocol import load_splits, retrieval_metrics
from shopee_match.features.image import rank_phash
from shopee_match.features.text import CharTfidfModel
from shopee_match.hashing import canonical_text_sha256, sha256_file
from shopee_match.reproducibility import seed_everything
from shopee_match.retrieval.benchmark import _profile_index, load_phase6_model
from shopee_match.retrieval.hybrid import reciprocal_rank_fusion
from shopee_match.retrieval.hybrid_benchmark import _ranking_jsonl, _truncate
from shopee_match.retrieval.vector_index import ExactCosineIndex, search_result_to_ranking
from shopee_match.training.multimodal_data import extract_frozen_multimodal_split
from shopee_match.training.multimodal_trainer import _git_state, _resolve_device

LOGGER = logging.getLogger(__name__)


def _existing_outputs(artifacts: HybridSystemArtifacts) -> list[str]:
    return [
        str(path)
        for path in (
            artifacts.access_marker,
            artifacts.embeddings,
            artifacts.ranking,
            artifacts.scored_pairs,
            artifacts.assignments,
            artifacts.metrics,
            artifacts.review,
            artifacts.report,
        )
        if path.exists()
    ]


def preflight_hybrid_system_evaluation(config_path: Path) -> dict[str, object]:
    """Validate frozen sources and guards without reading held-out rows."""
    config = load_hybrid_system_evaluation_config(config_path)
    commit, dirty = _git_state()
    outputs = _existing_outputs(config.artifacts)
    device = _resolve_device(config.runtime.device)
    return {
        "status": "ready" if not dirty and not outputs else "blocked",
        "test_accessed_by_preflight": False,
        "prior_version_test_results_exist": Path(
            "artifacts/final_evaluation/system_test/metrics.json"
        ).exists(),
        "version_access_marker_exists": config.artifacts.access_marker.exists(),
        "git_commit": commit,
        "git_dirty": dirty,
        "device": str(device),
        "existing_outputs": outputs,
        "candidate_k": config.policy.candidate_k,
        "entity_config_sha256": config.source.entity_config_sha256,
        "entity_metrics_sha256": config.source.entity_metrics_sha256,
    }


def _render_report(run: dict[str, Any]) -> str:
    validation = run["validation_reference"]
    retrieval = run["test"]["retrieval"]
    cluster = run["test"]["clustering"]
    pair = run["test"]["candidate_pair_classification"]
    edge = run["test"]["accepted_edge_metrics"]
    policy = run["frozen_policy"]
    retrieval_rows = "\n".join(
        f"| mAP@{k} | {validation['retrieval'][str(k)]['map@' + str(k)]:.5f} | "
        f"{retrieval[str(k)]['map@' + str(k)]:.5f} |\n"
        f"| Recall@{k} | {validation['retrieval'][str(k)]['recall@' + str(k)]:.5f} | "
        f"{retrieval[str(k)]['recall@' + str(k)]:.5f} |"
        for k in run["evaluation"]["metric_k_values"]
    )
    validation_cluster = validation["clustering"]
    clustering_rows = "\n".join(
        (
            f"| Pairwise precision | {validation_cluster['pairwise']['precision']:.5f} | "
            f"{cluster['pairwise']['precision']:.5f} |",
            f"| Pairwise recall | {validation_cluster['pairwise']['recall']:.5f} | "
            f"{cluster['pairwise']['recall']:.5f} |",
            f"| Pairwise F1 | {validation_cluster['pairwise']['f1']:.5f} | "
            f"{cluster['pairwise']['f1']:.5f} |",
            f"| B-cubed F1 | {validation_cluster['b_cubed']['f1']:.5f} | "
            f"{cluster['b_cubed']['f1']:.5f} |",
            f"| False-merge pair rate | {validation_cluster['false_merge_pair_rate']:.5f} | "
            f"{cluster['false_merge_pair_rate']:.5f} |",
            f"| False-split group rate | {validation_cluster['false_split_group_rate']:.5f} | "
            f"{cluster['false_split_group_rate']:.5f} |",
        )
    )
    attachment = policy["singleton_attachment"]
    return f"""# Hybrid System Confirmatory Evaluation

Status: **{run["status"]}**. This version uses the validation-selected hybrid candidate union and
supported singleton attachment without changing the frozen encoders or pair head.

## Frozen policy

- Commit: `{run["provenance"]["git_commit"]}` (`git_dirty=false`)
- Config SHA-256: `{run["provenance"]["config_sha256"]}`
- Hybrid entity config SHA-256: `{run["provenance"]["entity_config_sha256"]}`
- Hybrid entity metrics SHA-256: `{run["provenance"]["entity_metrics_sha256"]}`
- Candidate K / core threshold / reciprocal rank: `{policy["candidate_k"]}` /
  `{policy["pair_probability_threshold"]:.2f}` / `{policy["reciprocal_rank"]}`
- Singleton attachment threshold / rank / support: `{attachment["probability_threshold"]:.2f}` /
  `{attachment["reciprocal_rank"]}` / `{attachment["minimum_support"]}`

## Retrieval: validation to test

| Metric | Validation | Test |
|---|---:|---:|
{retrieval_rows}

## Pair and entity results

| Pair metric | Test |
|---|---:|
| Candidate-conditioned precision | {pair["precision"]:.5f} |
| Candidate-conditioned recall | {pair["recall_within_candidates"]:.5f} |
| Candidate-conditioned F1 | {pair["f1_within_candidates"]:.5f} |
| PR-AUC | {pair["average_precision_pr_auc"]:.5f} |
| Accepted-edge precision | {edge["precision"]:.5f} |
| Accepted-edge global recall | {edge["recall"]:.5f} |

| Entity metric | Validation | Test |
|---|---:|---:|
{clustering_rows}

## Interpretation

This is a confirmatory test of a new validation-frozen system version. The same test split was
previously used to report the predecessor system, so this result is not described as globally
unseen. No threshold, candidate K, fusion weight, or graph rule is selected from this output.

The access marker and immutable output paths block accidental reruns of this version.
"""


def run_hybrid_system_evaluation(config_path: Path) -> dict[str, object]:
    """Run the versioned hybrid system on test once, without test-time selection."""
    config = load_hybrid_system_evaluation_config(config_path)
    existing = _existing_outputs(config.artifacts)
    if existing:
        raise OutputConflictError(
            "Hybrid test access or output already exists; refusing to rerun: " + ", ".join(existing)
        )
    commit, dirty = _git_state()
    if dirty:
        raise DataValidationError("Hybrid system evaluation requires a clean Git worktree")
    config_sha = canonical_text_sha256(config.config_path)
    _write_text_atomic(
        config.artifacts.access_marker,
        json.dumps(
            {
                "status": "hybrid_system_test_access_started",
                "config_sha256": config_sha,
                "git_commit": commit,
                "git_dirty": False,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
    )

    started = time.perf_counter()
    seed_everything(config.seed, deterministic=True)
    device = _resolve_device(config.runtime.device)
    entity_config = config.source.entity_config
    hybrid_config = entity_config.source.hybrid
    phase7 = hybrid_config.source.experiment
    phase6 = phase7.source.experiment
    multimodal = phase6.source.experiment
    splits = load_splits(multimodal.data.metadata_csv, multimodal.data.split_manifest)
    test = splits["test"]

    LOGGER.info("Hybrid final stage 1/6: extracting frozen test embeddings")
    posting_ids, image, text, extraction = extract_frozen_multimodal_split(
        multimodal,
        "test",
        device=device,
        batch_size=config.runtime.embedding_batch_size,
        num_workers=config.runtime.num_workers,
    )
    if posting_ids != tuple(item.posting_id for item in test.items):
        raise DataValidationError("Hybrid test embeddings do not align with test manifest")
    model = load_phase6_model(phase7, device)
    embeddings, fusion_seconds = _joint_embeddings(
        model,
        image,
        text,
        device=device,
        batch_size=config.runtime.embedding_batch_size,
    )
    _write_embeddings_atomic(config.artifacts.embeddings, posting_ids, embeddings)

    LOGGER.info("Hybrid final stage 2/6: building dense, sparse-title, and pHash rankings")
    retrieval_started = time.perf_counter()
    exact = ExactCosineIndex(posting_ids, embeddings)
    indices, scores = exact.search(
        embeddings,
        hybrid_config.fusion.dense_candidate_k,
        query_ids=posting_ids,
        block_size=config.evaluation.exact_block_size,
    )
    dense = search_result_to_ranking(posting_ids, posting_ids, indices, scores)
    tfidf_started = time.perf_counter()
    tfidf_model = CharTfidfModel.fit(
        splits["train"].items,
        hybrid_config.tfidf.ngram_range,
        hybrid_config.tfidf.max_features,
    )
    tfidf = tfidf_model.rank(test.items, hybrid_config.tfidf.candidate_k)
    tfidf_seconds = time.perf_counter() - tfidf_started
    phash_started = time.perf_counter()
    phash = rank_phash(test.items, hybrid_config.fusion.phash_candidate_k)
    phash_seconds = time.perf_counter() - phash_started
    fusion_started = time.perf_counter()
    ranking = reciprocal_rank_fusion(
        {"dense": dense, "tfidf": tfidf, "phash": phash},
        weights={
            "dense": hybrid_config.fusion.dense_weight,
            "tfidf": hybrid_config.fusion.tfidf_weight,
            "phash": hybrid_config.fusion.phash_weight,
        },
        rrf_constant=hybrid_config.fusion.rrf_constant,
        top_k=config.policy.candidate_k,
    )
    rank_fusion_seconds = time.perf_counter() - fusion_started
    retrieval_seconds = time.perf_counter() - retrieval_started
    _write_text_atomic(config.artifacts.ranking, _ranking_jsonl(ranking))
    retrieval = {
        str(k): retrieval_metrics(_truncate(ranking, k), test.label_by_id, (k,), k)
        for k in config.evaluation.metric_k_values
    }
    search_profile = _profile_index(
        exact,
        embeddings,
        posting_ids,
        hybrid_config.fusion.dense_candidate_k,
        block_size=config.evaluation.exact_block_size,
        query_count=config.evaluation.latency_query_count,
        repetitions=config.evaluation.latency_repetitions,
    )

    LOGGER.info("Hybrid final stage 3/6: scoring frozen candidate pairs")
    pair_started = time.perf_counter()
    pairs = score_candidate_pairs(
        model,
        posting_ids,
        test.items,
        embeddings,
        ranking,
        device,
        batch_size=config.runtime.pair_batch_size,
    )
    pair_seconds = time.perf_counter() - pair_started
    pair_classification = candidate_pair_classification_metrics(
        pairs,
        test.label_by_id,
        threshold=config.policy.pair_probability_threshold,
        calibration_bins=config.evaluation.calibration_bins,
        required_recall=config.evaluation.required_recall,
        required_precision=config.evaluation.required_precision,
    )
    edge = edge_metrics(
        pairs,
        test.label_by_id,
        pair_probability_threshold=config.policy.pair_probability_threshold,
        reciprocal_rank=config.policy.reciprocal_rank,
        variant_conflict_override_probability=config.policy.variant_conflict_override_probability,
    )

    LOGGER.info("Hybrid final stage 4/6: applying frozen graph and singleton policy")
    attachment = config.policy.singleton_attachment
    assignments, graph = build_conservative_clusters(
        posting_ids,
        pairs,
        pair_probability_threshold=config.policy.pair_probability_threshold,
        reciprocal_rank=config.policy.reciprocal_rank,
        cross_component_minimum_coverage=config.policy.cross_component_minimum_coverage,
        variant_conflict_override_probability=config.policy.variant_conflict_override_probability,
        maximum_cluster_size=config.policy.maximum_cluster_size,
        manual_review_margin=config.policy.manual_review_margin,
        singleton_attachment=attachment.enabled,
        singleton_probability_threshold=attachment.probability_threshold,
        singleton_reciprocal_rank=attachment.reciprocal_rank,
        singleton_minimum_support=attachment.minimum_support,
        singleton_target_margin=attachment.target_margin,
    )
    cluster = clustering_metrics(assignments, test.label_by_id)
    strata = group_size_strata(assignments, test.label_by_id)
    review = _failure_review(
        test, assignments, example_limit=config.evaluation.failure_example_limit
    )

    LOGGER.info("Hybrid final stage 5/6: assembling metrics and disclosure")
    validation_metrics = config.source.entity_metrics
    validation_curve = entity_config.source.hybrid_metrics["hybrid"]["retrieval_curve"]
    validation_reference = {
        "retrieval": {str(k): validation_curve[str(k)] for k in config.evaluation.metric_k_values},
        "clustering": validation_metrics["selection"]["selected"]["clustering"],
    }
    wall_seconds = time.perf_counter() - started
    run: dict[str, Any] = {
        "pipeline_version": "hybrid.system_evaluation.v1",
        "status": "hybrid_system_confirmatory_test_complete",
        "provenance": {
            "config_sha256": config_sha,
            "entity_config_sha256": config.source.entity_config_sha256,
            "entity_metrics_sha256": config.source.entity_metrics_sha256,
            "hybrid_config_sha256": entity_config.source.hybrid_config_sha256,
            "hybrid_metrics_sha256": entity_config.source.hybrid_metrics_sha256,
            "phase6_checkpoint_sha256": phase7.source.checkpoint_sha256,
            "split_manifest_sha256": sha256_file(multimodal.data.split_manifest),
            "git_commit": commit,
            "git_dirty": False,
            "seed": config.seed,
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "device": str(device),
        },
        "data": {
            "split": "test",
            "listings": len(posting_ids),
            "test_accessed": True,
            "version_evaluation_count": 1,
            "predecessor_system_test_results_exist": True,
            "selection_on_test": False,
        },
        "evaluation": {"metric_k_values": list(config.evaluation.metric_k_values)},
        "frozen_policy": asdict(config.policy),
        "validation_reference": validation_reference,
        "test": {
            "retrieval": retrieval,
            "candidate_pair_classification": pair_classification,
            "accepted_edge_metrics": edge,
            "clustering": cluster,
            "group_size_strata": strata,
            "graph": asdict(graph),
        },
        "efficiency": {
            "image_extraction_seconds": extraction["image_extraction_seconds"],
            "text_extraction_seconds": extraction["text_extraction_seconds"],
            "joint_fusion_seconds": fusion_seconds,
            "hybrid_retrieval_seconds": retrieval_seconds,
            "tfidf_fit_and_rank_seconds": tfidf_seconds,
            "phash_ranking_seconds": phash_seconds,
            "rank_fusion_seconds": rank_fusion_seconds,
            "pair_scoring_seconds": pair_seconds,
            "pair_scoring_pairs_per_second": len(pairs) / pair_seconds,
            "dense_exact_search": search_profile,
            "wall_time_seconds": wall_seconds,
        },
        "model": {
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
            "joint_embedding_dimension": int(embeddings.shape[1]),
        },
        "failure_analysis": review["counts"],
        "artifacts": {
            "access_marker": str(config.artifacts.access_marker),
            "embeddings": str(config.artifacts.embeddings),
            "ranking": str(config.artifacts.ranking),
            "scored_pairs": str(config.artifacts.scored_pairs),
            "assignments": str(config.artifacts.assignments),
            "review": str(config.artifacts.review),
        },
    }

    LOGGER.info("Hybrid final stage 6/6: writing immutable confirmatory evidence")
    _write_text_atomic(
        config.artifacts.scored_pairs,
        "".join(json.dumps(scored_pair_payload(pair), sort_keys=True) + "\n" for pair in pairs),
    )
    _write_assignments_atomic(config.artifacts.assignments, assignments)
    _write_text_atomic(config.artifacts.review, json.dumps(review, indent=2, sort_keys=True) + "\n")
    _write_text_atomic(config.artifacts.metrics, json.dumps(run, indent=2, sort_keys=True) + "\n")
    _write_text_atomic(config.artifacts.report, _render_report(run))
    return {
        "status": run["status"],
        "metrics": str(config.artifacts.metrics),
        "report": str(config.artifacts.report),
        "test_accessed": True,
        "retrieval_recall": retrieval[str(config.policy.candidate_k)][
            f"recall@{config.policy.candidate_k}"
        ],
        "pairwise_precision": cluster["pairwise"]["precision"],
        "pairwise_recall": cluster["pairwise"]["recall"],
        "pairwise_f1": cluster["pairwise"]["f1"],
        "b_cubed_f1": cluster["b_cubed"]["f1"],
    }
