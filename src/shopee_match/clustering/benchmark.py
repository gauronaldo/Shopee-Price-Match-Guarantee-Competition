"""Phase 8 validation-only pair scoring and conservative entity-resolution benchmark."""

from __future__ import annotations

import csv
import json
import logging
import os
import platform
import time
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from numpy.typing import NDArray

from shopee_match.clustering.config import EntityResolutionConfig, load_entity_resolution_config
from shopee_match.clustering.graph import (
    ClusterAssignment,
    build_conservative_clusters,
    score_candidate_pairs,
    scored_pair_payload,
)
from shopee_match.clustering.metrics import clustering_metrics, edge_metrics, group_size_strata
from shopee_match.errors import DataValidationError, OutputConflictError
from shopee_match.evaluation.protocol import EvaluationSplit, load_named_split
from shopee_match.hashing import canonical_text_sha256, sha256_file
from shopee_match.reproducibility import seed_everything
from shopee_match.retrieval.benchmark import _load_phase6_model
from shopee_match.retrieval.vector_index import ExactCosineIndex, search_result_to_ranking
from shopee_match.training.multimodal_trainer import _git_state, _resolve_device

FloatArray = NDArray[np.float32]
LOGGER = logging.getLogger(__name__)


def _write_text_atomic(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(content, encoding="utf-8")
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


def _load_embeddings(config: EntityResolutionConfig) -> tuple[tuple[str, ...], FloatArray]:
    try:
        with np.load(config.source.embedding_cache_path, allow_pickle=False) as payload:
            posting_ids = tuple(str(value) for value in payload["posting_ids"].tolist())
            embeddings = payload["embeddings"].astype(np.float32, copy=False)
            contract = json.loads(str(payload["contract"][0]))
    except (OSError, KeyError, ValueError, json.JSONDecodeError) as exc:
        raise DataValidationError("Cannot load the frozen Phase 7 embedding cache") from exc
    phase7 = config.source.experiment
    if (
        contract.get("version") != "phase7.listing_embeddings.v1"
        or contract.get("split") != "validation"
        or contract.get("test_accessed") is not False
        or contract.get("normalized") is not True
        or contract.get("checkpoint_sha256") != phase7.source.checkpoint_sha256
        or embeddings.ndim != 2
        or embeddings.shape[0] != len(posting_ids)
        or not np.isfinite(embeddings).all()
        or not np.allclose(np.linalg.norm(embeddings, axis=1), 1.0, atol=1e-5)
    ):
        raise DataValidationError("Phase 7 embedding cache contract is invalid")
    return posting_ids, embeddings


def _selection_key(trial: dict[str, Any]) -> tuple[float, ...]:
    clustering = trial["clustering"]
    return (
        float(trial["passes_precision_gate"]),
        float(clustering["b_cubed"]["f1"]),
        float(clustering["pairwise"]["f1"]),
        float(clustering["pairwise"]["precision"]),
        -float(clustering["false_split_group_rate"]),
        float(trial["pair_probability_threshold"]),
        -float(trial["reciprocal_rank"]),
        float(trial["cross_component_minimum_coverage"]),
    )


def _failure_review(
    split: EvaluationSplit,
    assignments: list[ClusterAssignment],
    *,
    example_limit: int,
) -> dict[str, Any]:
    assignment_by_id = {row.posting_id: row for row in assignments}
    item_by_id = {item.posting_id: item for item in split.items}
    predicted: dict[str, list[str]] = defaultdict(list)
    truth: dict[str, list[str]] = defaultdict(list)
    for posting_id, assignment in assignment_by_id.items():
        predicted[assignment.entity_id].append(posting_id)
        truth[split.label_by_id[posting_id]].append(posting_id)

    false_merges: list[dict[str, Any]] = []
    for entity_id, members in predicted.items():
        labels = sorted({split.label_by_id[posting_id] for posting_id in members})
        if len(labels) <= 1:
            continue
        exemplar = assignment_by_id[members[0]]
        false_merges.append(
            {
                "entity_id": entity_id,
                "cluster_size": len(members),
                "label_count": len(labels),
                "cluster_confidence": exemplar.cluster_confidence,
                "manual_review": exemplar.manual_review,
                "members": [
                    {
                        "posting_id": posting_id,
                        "label_group": split.label_by_id[posting_id],
                        "title": item_by_id[posting_id].title,
                    }
                    for posting_id in sorted(members)[:8]
                ],
            }
        )
    false_merges.sort(key=lambda row: (-int(row["cluster_size"]), float(row["cluster_confidence"])))

    false_splits: list[dict[str, Any]] = []
    for label, members in truth.items():
        entities = sorted({assignment_by_id[posting_id].entity_id for posting_id in members})
        if len(entities) <= 1:
            continue
        false_splits.append(
            {
                "label_group": label,
                "group_size": len(members),
                "predicted_fragments": len(entities),
                "members": [
                    {
                        "posting_id": posting_id,
                        "entity_id": assignment_by_id[posting_id].entity_id,
                        "title": item_by_id[posting_id].title,
                    }
                    for posting_id in sorted(members)[:8]
                ],
            }
        )
    false_splits.sort(key=lambda row: (-int(row["predicted_fragments"]), -int(row["group_size"])))
    review_by_entity: dict[str, dict[str, Any]] = {
        row.entity_id: {
            "entity_id": row.entity_id,
            "cluster_size": row.cluster_size,
            "cluster_confidence": row.cluster_confidence,
        }
        for row in assignments
        if row.manual_review
    }
    review_clusters = sorted(
        review_by_entity.values(),
        key=lambda row: (
            float(row["cluster_confidence"]),
            -int(row["cluster_size"]),
            str(row["entity_id"]),
        ),
    )
    return {
        "false_merge_examples": false_merges[:example_limit],
        "false_split_examples": false_splits[:example_limit],
        "manual_review_examples": review_clusters[:example_limit],
        "counts": {
            "impure_clusters": len(false_merges),
            "split_label_groups": len(false_splits),
            "manual_review_clusters": len(review_clusters),
        },
        "labels_used_for_analysis_only": True,
    }


def _render_report(run: dict[str, Any]) -> str:
    selected = run["selection"]["selected"]
    cluster = selected["clustering"]
    pairwise = cluster["pairwise"]
    b3 = cluster["b_cubed"]
    edge = selected["edge"]
    graph = selected["graph"]
    variant_override = run["selection"]["variant_conflict_override_probability"]
    cluster_count = cluster["clusters"]
    singleton_count = cluster["singleton_clusters"]
    pairwise_row = (
        f"| Pairwise clusters | {pairwise['precision']:.5f} | "
        f"{pairwise['recall']:.5f} | {pairwise['f1']:.5f} |"
    )
    strata_rows = "\n".join(
        f"| {band} | {row['groups']:.0f} | {row['unsplit_group_rate']:.5f} | "
        f"{row['mean_predicted_fragments']:.3f} |"
        for band, row in run["group_size_strata"].items()
    )
    return f"""# Entity Resolution Benchmark

Phase 8 status: **{run["status"]}**. Thresholds and graph rules are selected on validation only;
test remains untouched. Ground-truth labels are used for selection and analysis, never as graph
features or edge-construction inputs.

## Frozen inputs

- Listings: `{run["data"]["listings"]:,}` validation listings
- Candidate budget: Top-`{run["source"]["candidate_k"]}` exact cosine neighbours
- Pair scorer: accepted Phase 6 symmetric pair head
- Candidate Recall@50 ceiling: `{run["source"]["candidate_recall"]:.5f}`
- Test accessed: `false`

## Selected graph policy

| Setting | Selected value |
|---|---:|
| Pair probability threshold | {selected["pair_probability_threshold"]:.6f} |
| Reciprocal-neighbour rank | {selected["reciprocal_rank"]} |
| Cross-component coverage | {selected["cross_component_minimum_coverage"]:.2f} |
| Variant-conflict override probability | {variant_override:.2f} |
| Maximum cluster size | {run["selection"]["maximum_cluster_size"]} |

## Validation metrics

| Metric family | Precision | Recall | F1 |
|---|---:|---:|---:|
| Accepted candidate edges | {edge["precision"]:.5f} | {edge["recall"]:.5f} | {edge["f1"]:.5f} |
{pairwise_row}
| B-cubed clustering | {b3["precision"]:.5f} | {b3["recall"]:.5f} | {b3["f1"]:.5f} |

- False-merge pair rate: `{cluster["false_merge_pair_rate"]:.5f}`
- Impure non-singleton cluster rate: `{cluster["impure_non_singleton_cluster_rate"]:.5f}`
- False-split group rate: `{cluster["false_split_group_rate"]:.5f}`
- Predicted clusters / singleton clusters: `{cluster_count:.0f}` / `{singleton_count:.0f}`
- Maximum predicted cluster size: `{cluster["maximum_cluster_size"]:.0f}`
- Manual-review clusters: `{cluster["manual_review_clusters"]:.0f}`

## Graph audit

| Counter | Value |
|---|---:|
| Unique Top-K candidate pairs | {graph["candidate_pairs"]:,} |
| Eligible reciprocal edges | {graph["eligible_edges"]:,} |
| Accepted component merges | {graph["accepted_merges"]:,} |
| Rejected by cluster-size cap | {graph["size_rejections"]:,} |
| Rejected by transitive consistency | {graph["consistency_rejections"]:,} |
| Rejected variant conflicts | {graph["variant_conflict_rejected"]:,} |

## Performance by true group size

| Group size | Groups | Unsplit-group rate | Mean predicted fragments |
|---|---:|---:|---:|
{strata_rows}

## Interpretation

Pairwise precision is the primary false-merge safety metric because one false edge can merge
otherwise correct components. B-cubed F1 balances entity purity and fragmentation per listing.
The reciprocal-neighbour rule removes one-sided retrieval coincidences; the cross-component
coverage rule blocks a single bridge from joining two established components unless enough members
support the merge. Variant-conflicting titles require a higher pair probability.

This is a validation-selected operating point, not a final test claim. Detailed false-merge,
false-split, and manual-review examples remain in the ignored Phase 8 review artifact.

Manual inspection shows two dominant categories: same-brand or same-package variants can still
form false-merge bridges, while large groups with diverse images and titles are fragmented by the
strict reciprocal and full cross-component-support rules. Some near-identical cross-label examples
also remain plausible label ambiguities and are documented rather than relabeled.

## Reproduction

```powershell
.venv\\Scripts\\shopee-entity-resolution benchmark `
  --config configs\\experiment\\entity_resolution_benchmark.yaml
```
"""


def run_entity_resolution_benchmark(config_path: Path) -> dict[str, object]:
    """Score frozen Phase 7 candidates and select a conservative validation graph policy."""
    config = load_entity_resolution_config(config_path)
    existing = [
        str(path) for path in (config.artifacts.metrics, config.artifacts.report) if path.exists()
    ]
    if existing:
        raise OutputConflictError(
            "Refusing to overwrite completed Phase 8 evidence: " + ", ".join(existing)
        )
    started = time.perf_counter()
    seed_everything(config.seed, deterministic=True)
    device = _resolve_device(config.pair_scoring.device)
    LOGGER.info("Phase 8 stage 1/4: loading frozen validation embeddings on %s", device)
    phase7 = config.source.experiment
    source_experiment = phase7.source.experiment.source.experiment
    split = load_named_split(
        source_experiment.data.metadata_csv,
        source_experiment.data.split_manifest,
        "validation",
    )
    posting_ids, embeddings = _load_embeddings(config)
    expected_ids = tuple(item.posting_id for item in split.items)
    if posting_ids != expected_ids:
        raise DataValidationError("Phase 7 embeddings do not align with the validation manifest")
    candidate_k = int(config.source.metrics["selection"]["candidate_k"])
    exact = ExactCosineIndex(posting_ids, embeddings)
    indices, scores = exact.search(
        embeddings,
        candidate_k,
        query_ids=posting_ids,
        block_size=phase7.exact.block_size,
    )
    ranking = search_result_to_ranking(posting_ids, posting_ids, indices, scores)
    model = _load_phase6_model(phase7, device)
    LOGGER.info("Phase 8 stage 2/4: scoring unique pairs from Top-%d candidates", candidate_k)
    scoring_started = time.perf_counter()
    pairs = score_candidate_pairs(
        model,
        posting_ids,
        split.items,
        embeddings,
        ranking,
        device,
        batch_size=config.pair_scoring.batch_size,
    )
    scoring_seconds = time.perf_counter() - scoring_started
    LOGGER.info(
        "Scored %d unique pairs in %.2fs; starting graph-policy sweep",
        len(pairs),
        scoring_seconds,
    )

    trials: list[dict[str, Any]] = []
    selected_assignments: list[ClusterAssignment] | None = None
    selected_trial: dict[str, Any] | None = None
    for threshold in config.selection.pair_probability_thresholds:
        LOGGER.info(
            "Phase 8 stage 3/4: evaluating pair threshold %.3f",
            threshold,
        )
        for reciprocal_rank in config.selection.reciprocal_rank_values:
            for coverage in config.selection.cross_component_coverage_values:
                assignments, diagnostics = build_conservative_clusters(
                    posting_ids,
                    pairs,
                    pair_probability_threshold=threshold,
                    reciprocal_rank=reciprocal_rank,
                    cross_component_minimum_coverage=coverage,
                    variant_conflict_override_probability=(
                        config.selection.variant_conflict_override_probability
                    ),
                    maximum_cluster_size=config.selection.maximum_cluster_size,
                    manual_review_margin=config.selection.manual_review_margin,
                )
                cluster = clustering_metrics(assignments, split.label_by_id)
                edge = edge_metrics(
                    pairs,
                    split.label_by_id,
                    pair_probability_threshold=threshold,
                    reciprocal_rank=reciprocal_rank,
                    variant_conflict_override_probability=(
                        config.selection.variant_conflict_override_probability
                    ),
                )
                trial: dict[str, Any] = {
                    "pair_probability_threshold": threshold,
                    "reciprocal_rank": reciprocal_rank,
                    "cross_component_minimum_coverage": coverage,
                    "passes_precision_gate": (
                        cluster["pairwise"]["precision"]
                        >= config.selection.minimum_pairwise_precision
                    ),
                    "edge": edge,
                    "clustering": cluster,
                    "graph": asdict(diagnostics),
                }
                trials.append(trial)
                if selected_trial is None or _selection_key(trial) > _selection_key(selected_trial):
                    selected_trial = trial
                    selected_assignments = assignments
    if selected_trial is None or selected_assignments is None:
        raise DataValidationError("Phase 8 selection grid produced no trials")

    review = _failure_review(
        split,
        selected_assignments,
        example_limit=config.selection.failure_example_limit,
    )
    commit, dirty = _git_state()
    status = (
        "phase8_complete_validation_only"
        if selected_trial["passes_precision_gate"]
        else "phase8_exit_gate_failed"
    )
    run: dict[str, Any] = {
        "pipeline_version": "phase8.entity_resolution.v1",
        "status": status,
        "provenance": {
            "config_sha256": canonical_text_sha256(config.config_path),
            "phase7_config_sha256": config.source.phase7_config_sha256,
            "phase7_metrics_sha256": config.source.phase7_metrics_sha256,
            "embedding_cache_sha256": config.source.embedding_cache_sha256,
            "phase6_checkpoint_sha256": phase7.source.checkpoint_sha256,
            "split_manifest_sha256": sha256_file(source_experiment.data.split_manifest),
            "git_commit": commit,
            "git_dirty": dirty,
            "seed": config.seed,
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "device": str(device),
        },
        "data": {"split": "validation", "listings": len(posting_ids), "test_accessed": False},
        "source": {
            "candidate_k": candidate_k,
            "candidate_recall": config.source.metrics["exact"]["selected_recall"],
            "pair_scorer": "phase6_symmetric_pair_head",
        },
        "pair_scoring": {
            "unique_pairs": len(pairs),
            "seconds": scoring_seconds,
            "pairs_per_second": len(pairs) / scoring_seconds,
            "batch_size": config.pair_scoring.batch_size,
        },
        "selection": {
            "objective": "maximum_b_cubed_f1_subject_to_pairwise_precision_gate",
            "minimum_pairwise_precision": config.selection.minimum_pairwise_precision,
            "variant_conflict_override_probability": (
                config.selection.variant_conflict_override_probability
            ),
            "maximum_cluster_size": config.selection.maximum_cluster_size,
            "selected": selected_trial,
            "trials": trials,
        },
        "group_size_strata": group_size_strata(selected_assignments, split.label_by_id),
        "failure_analysis": review["counts"],
        "runtime_seconds": time.perf_counter() - started,
        "artifacts": {
            "scored_pairs": str(config.artifacts.scored_pairs),
            "assignments": str(config.artifacts.assignments),
            "review": str(config.artifacts.review),
        },
        "test": {"status": "disabled_phase8_validation_only"},
    }
    LOGGER.info("Phase 8 stage 4/4: writing immutable metrics and review artifacts")
    _write_text_atomic(
        config.artifacts.scored_pairs,
        "".join(json.dumps(scored_pair_payload(pair), sort_keys=True) + "\n" for pair in pairs),
    )
    _write_assignments_atomic(config.artifacts.assignments, selected_assignments)
    _write_text_atomic(config.artifacts.review, json.dumps(review, indent=2, sort_keys=True) + "\n")
    _write_text_atomic(config.artifacts.metrics, json.dumps(run, indent=2, sort_keys=True) + "\n")
    _write_text_atomic(config.artifacts.report, _render_report(run))
    cluster = cast(dict[str, Any], selected_trial["clustering"])
    return {
        "status": status,
        "pairwise_f1": cluster["pairwise"]["f1"],
        "b_cubed_f1": cluster["b_cubed"]["f1"],
        "false_merge_pair_rate": cluster["false_merge_pair_rate"],
        "false_split_group_rate": cluster["false_split_group_rate"],
        "metrics": str(config.artifacts.metrics),
        "report": str(config.artifacts.report),
        "test_accessed": False,
    }
