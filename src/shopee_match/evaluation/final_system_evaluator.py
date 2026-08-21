"""One-time end-to-end held-out evaluation of the frozen catalog matching system."""

from __future__ import annotations

import csv
import json
import logging
import os
import platform
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from numpy.typing import NDArray

from shopee_match.clustering.benchmark import _failure_review
from shopee_match.clustering.graph import (
    ClusterAssignment,
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
from shopee_match.evaluation.final_system_config import (
    FinalArtifactConfig,
    load_final_system_evaluation_config,
)
from shopee_match.evaluation.protocol import load_named_split, retrieval_metrics
from shopee_match.hashing import canonical_text_sha256, sha256_file
from shopee_match.models import LearnedMultimodalFusion
from shopee_match.reproducibility import seed_everything
from shopee_match.retrieval.benchmark import _load_phase6_model, _profile_index
from shopee_match.retrieval.vector_index import ExactCosineIndex, search_result_to_ranking
from shopee_match.training.multimodal_data import extract_frozen_multimodal_split
from shopee_match.training.multimodal_trainer import _git_state, _resolve_device

LOGGER = logging.getLogger(__name__)
FloatArray = NDArray[np.float32]


def _write_text_atomic(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def _write_embeddings_atomic(
    path: Path, posting_ids: tuple[str, ...], embeddings: FloatArray
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            posting_ids=np.asarray(posting_ids, dtype=str),
            embeddings=embeddings,
        )
    temporary.replace(path)


def _write_assignments_atomic(path: Path, assignments: list[ClusterAssignment]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "posting_id",
                "entity_id",
                "cluster_size",
                "cluster_confidence",
                "manual_review",
            ),
        )
        writer.writeheader()
        writer.writerows(asdict(row) for row in assignments)
    temporary.replace(path)


def _existing_outputs(artifacts: FinalArtifactConfig) -> list[str]:
    return [
        str(path)
        for path in (
            artifacts.access_marker,
            artifacts.embeddings,
            artifacts.scored_pairs,
            artifacts.assignments,
            artifacts.metrics,
            artifacts.review,
            artifacts.report,
        )
        if path.exists()
    ]


def preflight_final_system_evaluation(config_path: Path) -> dict[str, object]:
    """Verify every frozen input and guard without loading held-out test rows."""
    config = load_final_system_evaluation_config(config_path)
    commit, dirty = _git_state()
    outputs = _existing_outputs(config.artifacts)
    device = _resolve_device(config.runtime.device)
    return {
        "status": "ready" if not dirty and not outputs else "blocked",
        "prior_test_access_detected": config.artifacts.access_marker.exists(),
        "test_accessed_by_preflight": False,
        "git_commit": commit,
        "git_dirty": dirty,
        "device": str(device),
        "existing_outputs": outputs,
        "candidate_k": config.policy.candidate_k,
        "pair_probability_threshold": config.policy.pair_probability_threshold,
        "reciprocal_rank": config.policy.reciprocal_rank,
        "cross_component_minimum_coverage": (config.policy.cross_component_minimum_coverage),
        "entity_config_sha256": config.source.entity_config_sha256,
        "entity_metrics_sha256": config.source.entity_metrics_sha256,
    }


def _joint_embeddings(
    model: LearnedMultimodalFusion,
    image: FloatArray,
    text: FloatArray,
    *,
    device: torch.device,
    batch_size: int,
) -> tuple[FloatArray, float]:
    chunks: list[FloatArray] = []
    started = time.perf_counter()
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(image), batch_size):
            image_batch = torch.from_numpy(image[start : start + batch_size]).to(device)
            text_batch = torch.from_numpy(text[start : start + batch_size]).to(device)
            chunks.append(
                cast(FloatArray, model(image_batch, text_batch).cpu().numpy()).astype(
                    np.float32, copy=False
                )
            )
    if not chunks:
        raise DataValidationError("Final evaluation cannot embed an empty split")
    embeddings = np.concatenate(chunks)
    if not np.isfinite(embeddings).all() or not np.allclose(
        np.linalg.norm(embeddings, axis=1), 1.0, atol=1e-5
    ):
        raise DataValidationError("Final joint embeddings are not finite and L2-normalized")
    return embeddings, time.perf_counter() - started


def _render_report(run: dict[str, Any]) -> str:
    validation = run["validation_reference"]
    test = run["test"]
    retrieval20 = test["retrieval"]["20"]
    retrieval50 = test["retrieval"]["50"]
    edge = test["accepted_edge_metrics"]
    pair = test["candidate_pair_classification"]
    cluster = test["clustering"]
    pairwise = cluster["pairwise"]
    b3 = cluster["b_cubed"]
    validation_cluster = validation["clustering"]
    retrieval_rows = "\n".join(
        (
            f"| mAP@20 | {validation['retrieval']['map@20']:.5f} | {retrieval20['map@20']:.5f} |",
            f"| Recall@20 | {validation['retrieval']['recall@20']:.5f} | "
            f"{retrieval20['recall@20']:.5f} |",
            f"| mAP@50 | {validation['retrieval']['map@50']:.5f} | {retrieval50['map@50']:.5f} |",
            f"| Recall@50 | {validation['retrieval']['recall@50']:.5f} | "
            f"{retrieval50['recall@50']:.5f} |",
        )
    )
    clustering_rows = "\n".join(
        (
            f"| Pairwise precision | {validation_cluster['pairwise']['precision']:.5f} | "
            f"{pairwise['precision']:.5f} |",
            f"| Pairwise recall | {validation_cluster['pairwise']['recall']:.5f} | "
            f"{pairwise['recall']:.5f} |",
            f"| Pairwise F1 | {validation_cluster['pairwise']['f1']:.5f} | {pairwise['f1']:.5f} |",
            f"| B-cubed precision | {validation_cluster['b_cubed']['precision']:.5f} | "
            f"{b3['precision']:.5f} |",
            f"| B-cubed recall | {validation_cluster['b_cubed']['recall']:.5f} | "
            f"{b3['recall']:.5f} |",
            f"| B-cubed F1 | {validation_cluster['b_cubed']['f1']:.5f} | {b3['f1']:.5f} |",
            f"| False-merge pair rate | "
            f"{validation_cluster['false_merge_pair_rate']:.5f} | "
            f"{cluster['false_merge_pair_rate']:.5f} |",
            f"| False-split group rate | "
            f"{validation_cluster['false_split_group_rate']:.5f} | "
            f"{cluster['false_split_group_rate']:.5f} |",
        )
    )
    policy = run["frozen_policy"]
    search = run["efficiency"]["exact_search"]
    return f"""# Final System Evaluation

Status: **{run["status"]}**. The complete custom system was evaluated once with the exact
validation-selected retrieval, pair-scoring, and entity-resolution policy. No parameter,
threshold, candidate K, or graph rule was selected on this test result.

## Frozen contract

- Source commit: `{run["provenance"]["git_commit"]}` (`git_dirty=false`)
- Final config SHA-256: `{run["provenance"]["config_sha256"]}`
- Entity config SHA-256: `{run["provenance"]["entity_config_sha256"]}`
- Entity metrics SHA-256: `{run["provenance"]["entity_metrics_sha256"]}`
- Phase 6 checkpoint SHA-256: `{run["provenance"]["phase6_checkpoint_sha256"]}`
- Split manifest SHA-256: `{run["provenance"]["split_manifest_sha256"]}`
- Candidate K / pair threshold / reciprocal rank: `{policy["candidate_k"]}` /
  `{policy["pair_probability_threshold"]:.2f}` / `{policy["reciprocal_rank"]}`
- Cross-component coverage / maximum cluster size:
  `{policy["cross_component_minimum_coverage"]:.2f}` / `{policy["maximum_cluster_size"]}`

## Retrieval: validation to test

| Metric | Validation | Test |
|---|---:|---:|
{retrieval_rows}

## Pair decisions on retrieved candidates

| Metric | Test value |
|---|---:|
| Raw pair-head precision | {pair["precision"]:.5f} |
| Raw pair-head recall within candidates | {pair["recall_within_candidates"]:.5f} |
| Raw pair-head F1 within candidates | {pair["f1_within_candidates"]:.5f} |
| Average precision / PR-AUC | {pair["average_precision_pr_auc"]:.5f} |
| Brier score | {pair["brier_score"]:.5f} |
| Expected calibration error | {pair["expected_calibration_error"]:.5f} |
| Accepted reciprocal-edge precision | {edge["precision"]:.5f} |
| Accepted reciprocal-edge global recall | {edge["recall"]:.5f} |
| Accepted reciprocal-edge F1 | {edge["f1"]:.5f} |

Raw pair-head metrics are candidate-conditioned. Accepted-edge recall uses every true test pair as
its denominator and therefore includes retrieval and reciprocal-gating misses.

## Entity resolution: validation to test

| Metric | Validation | Test |
|---|---:|---:|
{clustering_rows}

## Efficiency

| Stage | Measured result |
|---|---:|
| Image extraction | {run["efficiency"]["image_throughput_per_second"]:.2f} listings/s |
| Text extraction | {run["efficiency"]["text_throughput_per_second"]:.2f} listings/s |
| Joint fusion | {run["efficiency"]["fusion_throughput_per_second"]:.2f} listings/s |
| Pair scoring | {run["efficiency"]["pair_scoring_pairs_per_second"]:.2f} pairs/s |
| Exact query p50 / p95 | {search["single_query_p50_ms"]:.3f} /
  {search["single_query_p95_ms"]:.3f} ms |
| End-to-end wall time | {run["efficiency"]["wall_time_seconds"]:.2f} s |

## Interpretation and disclosure

This is the first evaluation of the complete retrieval-plus-pair-plus-clustering system on the
held-out split. Earlier phases already reported component-level image, text, and multimodal test
results on the same frozen split; therefore it is held out from system-policy selection, but it is
not globally unseen to the project owner. The final test result is descriptive and is not used to
revise the operating point.

Detailed false-merge, false-split, and review examples remain in the ignored local artifact.
Aggregate failure counts and group-size strata are retained in the metrics JSON.

## Reproduction guard

```powershell
.venv\\Scripts\\shopee-final preflight `
  --config configs\\experiment\\final_system_evaluation.yaml
.venv\\Scripts\\shopee-final evaluate `
  --config configs\\experiment\\final_system_evaluation.yaml
```

The access marker and immutable outputs intentionally block a second evaluation.
"""


def run_final_system_evaluation(config_path: Path) -> dict[str, object]:
    """Evaluate the frozen end-to-end system on test exactly once."""
    config = load_final_system_evaluation_config(config_path)
    existing = _existing_outputs(config.artifacts)
    if existing:
        raise OutputConflictError(
            "Final test access or output already exists; refusing to rerun: " + ", ".join(existing)
        )
    commit, dirty = _git_state()
    if dirty:
        raise DataValidationError("Final system evaluation requires a clean Git worktree")
    config_sha = canonical_text_sha256(config.config_path)
    _write_text_atomic(
        config.artifacts.access_marker,
        json.dumps(
            {
                "status": "test_access_started",
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
    phase7 = entity_config.source.experiment
    multimodal = phase7.source.experiment.source.experiment
    LOGGER.info("Final stage 1/5: loading held-out split after immutable access marker")
    split = load_named_split(multimodal.data.metadata_csv, multimodal.data.split_manifest, "test")
    posting_ids, image, text, extraction = extract_frozen_multimodal_split(
        multimodal,
        "test",
        device=device,
        batch_size=config.runtime.embedding_batch_size,
        num_workers=config.runtime.num_workers,
    )
    expected_ids = tuple(item.posting_id for item in split.items)
    if posting_ids != expected_ids:
        raise DataValidationError("Final extracted embeddings do not align with test manifest")

    model = _load_phase6_model(phase7, device)
    embeddings, fusion_seconds = _joint_embeddings(
        model,
        image,
        text,
        device=device,
        batch_size=config.runtime.embedding_batch_size,
    )
    _write_embeddings_atomic(config.artifacts.embeddings, posting_ids, embeddings)

    LOGGER.info("Final stage 2/5: exact Top-%d candidate retrieval", config.policy.candidate_k)
    index = ExactCosineIndex(posting_ids, embeddings)
    indices, scores = index.search(
        embeddings,
        config.policy.candidate_k,
        query_ids=posting_ids,
        block_size=config.evaluation.exact_block_size,
    )
    ranking = search_result_to_ranking(posting_ids, posting_ids, indices, scores)
    retrieval = {
        str(k): retrieval_metrics(ranking, split.label_by_id, config.evaluation.recall_at, k)
        for k in config.evaluation.average_precision_at
    }
    search_profile = _profile_index(
        index,
        embeddings,
        posting_ids,
        config.policy.candidate_k,
        block_size=config.evaluation.exact_block_size,
        query_count=config.evaluation.latency_query_count,
        repetitions=config.evaluation.latency_repetitions,
    )

    LOGGER.info("Final stage 3/5: scoring frozen candidate pairs")
    pair_started = time.perf_counter()
    pairs = score_candidate_pairs(
        model,
        posting_ids,
        split.items,
        embeddings,
        ranking,
        device,
        batch_size=config.runtime.pair_batch_size,
    )
    pair_seconds = time.perf_counter() - pair_started
    pair_classification = candidate_pair_classification_metrics(
        pairs,
        split.label_by_id,
        threshold=config.policy.pair_probability_threshold,
        calibration_bins=config.evaluation.calibration_bins,
        required_recall=config.evaluation.required_recall,
        required_precision=config.evaluation.required_precision,
    )
    accepted_edge = edge_metrics(
        pairs,
        split.label_by_id,
        pair_probability_threshold=config.policy.pair_probability_threshold,
        reciprocal_rank=config.policy.reciprocal_rank,
        variant_conflict_override_probability=(config.policy.variant_conflict_override_probability),
    )

    LOGGER.info("Final stage 4/5: applying frozen conservative graph policy")
    assignments, graph = build_conservative_clusters(
        posting_ids,
        pairs,
        pair_probability_threshold=config.policy.pair_probability_threshold,
        reciprocal_rank=config.policy.reciprocal_rank,
        cross_component_minimum_coverage=config.policy.cross_component_minimum_coverage,
        variant_conflict_override_probability=(config.policy.variant_conflict_override_probability),
        maximum_cluster_size=config.policy.maximum_cluster_size,
        manual_review_margin=config.policy.manual_review_margin,
    )
    cluster = clustering_metrics(assignments, split.label_by_id)
    strata = group_size_strata(assignments, split.label_by_id)
    review = _failure_review(
        split, assignments, example_limit=config.evaluation.failure_example_limit
    )
    validation_selected = config.source.entity_metrics["selection"]["selected"]
    phase7_curve = entity_config.source.metrics["exact"]["retrieval_curve"]
    validation_reference = {
        "retrieval": {
            "map@20": phase7_curve["20"]["map@20"],
            "recall@20": phase7_curve["20"]["recall@20"],
            "map@50": phase7_curve["50"]["map@50"],
            "recall@50": phase7_curve["50"]["recall@50"],
        },
        "accepted_edge_metrics": validation_selected["edge"],
        "clustering": validation_selected["clustering"],
    }
    wall_seconds = time.perf_counter() - started
    policy_payload = asdict(config.policy)
    run: dict[str, Any] = {
        "pipeline_version": "final.system_evaluation.v1",
        "status": "final_system_test_complete",
        "provenance": {
            "config_sha256": config_sha,
            "entity_config_sha256": config.source.entity_config_sha256,
            "entity_metrics_sha256": config.source.entity_metrics_sha256,
            "phase7_config_sha256": entity_config.source.phase7_config_sha256,
            "phase7_metrics_sha256": entity_config.source.phase7_metrics_sha256,
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
            "final_system_evaluation_count": 1,
            "component_test_results_previously_reported": True,
            "selection_on_test": False,
        },
        "frozen_policy": policy_payload,
        "validation_reference": validation_reference,
        "test": {
            "retrieval": retrieval,
            "candidate_pair_classification": pair_classification,
            "accepted_edge_metrics": accepted_edge,
            "clustering": cluster,
            "group_size_strata": strata,
            "graph": asdict(graph),
        },
        "efficiency": {
            "image_extraction_seconds": extraction["image_extraction_seconds"],
            "image_throughput_per_second": (
                len(posting_ids) / extraction["image_extraction_seconds"]
            ),
            "text_extraction_seconds": extraction["text_extraction_seconds"],
            "text_throughput_per_second": (
                len(posting_ids) / extraction["text_extraction_seconds"]
            ),
            "fusion_seconds": fusion_seconds,
            "fusion_throughput_per_second": len(posting_ids) / fusion_seconds,
            "pair_scoring_seconds": pair_seconds,
            "pair_scoring_pairs_per_second": len(pairs) / pair_seconds,
            "exact_search": search_profile,
            "embedding_storage_bytes": int(embeddings.nbytes),
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
            "scored_pairs": str(config.artifacts.scored_pairs),
            "assignments": str(config.artifacts.assignments),
            "review": str(config.artifacts.review),
        },
    }

    LOGGER.info("Final stage 5/5: writing immutable final evidence")
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
        "retrieval_map@20": retrieval["20"]["map@20"],
        "pairwise_f1": cluster["pairwise"]["f1"],
        "b_cubed_f1": cluster["b_cubed"]["f1"],
    }
