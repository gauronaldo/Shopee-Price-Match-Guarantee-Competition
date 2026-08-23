"""Downstream validation of hybrid candidates with frozen pair scoring."""

from __future__ import annotations

import json
import logging
import math
import platform
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

import numpy as np

from shopee_match.clustering.benchmark import (
    _failure_review,
    _write_assignments_atomic,
    _write_text_atomic,
)
from shopee_match.clustering.graph import score_candidate_pairs, scored_pair_payload
from shopee_match.clustering.hybrid_entity_config import load_hybrid_entity_config
from shopee_match.clustering.recall_recovery import sweep_recovery_policies
from shopee_match.errors import DataValidationError, OutputConflictError
from shopee_match.evaluation.protocol import Ranking, ScoredCandidate, load_splits
from shopee_match.hashing import canonical_text_sha256, sha256_file
from shopee_match.reproducibility import seed_everything
from shopee_match.retrieval.benchmark import load_phase6_model
from shopee_match.retrieval.hybrid_benchmark import _load_embeddings
from shopee_match.training.multimodal_trainer import _git_state, _resolve_device

LOGGER = logging.getLogger(__name__)


def _load_ranking(path: Path, posting_ids: tuple[str, ...], *, expected_k: int) -> Ranking:
    valid_ids = set(posting_ids)
    result: Ranking = {}
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                payload = cast(dict[str, Any], json.loads(line))
                query_id = str(payload["query_posting_id"])
                rows = cast(list[dict[str, Any]], payload["candidates"])
                if (
                    query_id not in valid_ids
                    or query_id in result
                    or not 0 < len(rows) <= expected_k
                ):
                    raise DataValidationError(f"Invalid hybrid ranking row {line_number}")
                seen: set[str] = set()
                candidates: list[ScoredCandidate] = []
                for row in rows:
                    candidate_id = str(row["posting_id"])
                    score = float(row["score"])
                    if (
                        candidate_id not in valid_ids
                        or candidate_id == query_id
                        or candidate_id in seen
                        or not math.isfinite(score)
                    ):
                        raise DataValidationError(f"Invalid hybrid candidate at row {line_number}")
                    seen.add(candidate_id)
                    candidates.append(ScoredCandidate(candidate_id, score))
                result[query_id] = candidates
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        raise DataValidationError("Cannot load frozen hybrid candidate ranking") from exc
    if set(result) != valid_ids:
        raise DataValidationError("Hybrid ranking queries do not match validation IDs")
    return result


def _render_report(run: dict[str, Any]) -> str:
    incumbent = run["comparison"]["dense_incumbent"]
    selected = run["selection"]["selected"]
    cluster = selected["clustering"]
    pairwise = cluster["pairwise"]
    false_merge_row = (
        f"| False-merge pair rate | {incumbent['false_merge_pair_rate']:.5f} | "
        f"{cluster['false_merge_pair_rate']:.5f} |"
    )
    false_split_row = (
        f"| False-split group rate | {incumbent['false_split_group_rate']:.5f} | "
        f"{cluster['false_split_group_rate']:.5f} |"
    )
    return f"""# Hybrid Candidate Entity Resolution

Status: **{run["status"]}**. The experiment replaces only candidate generation; the accepted
multimodal embeddings and Phase 6 pair head remain frozen. Selection uses validation only.

| Clustering metric | Dense incumbent | Hybrid candidates |
|---|---:|---:|
| Pairwise precision | {incumbent["pairwise"]["precision"]:.5f} | {pairwise["precision"]:.5f} |
| Pairwise recall | {incumbent["pairwise"]["recall"]:.5f} | {pairwise["recall"]:.5f} |
| Pairwise F1 | {incumbent["pairwise"]["f1"]:.5f} | {pairwise["f1"]:.5f} |
| B-cubed F1 | {incumbent["b_cubed"]["f1"]:.5f} | {cluster["b_cubed"]["f1"]:.5f} |
{false_merge_row}
{false_split_row}

Hybrid candidate Recall@{run["source"]["candidate_k"]} is
`{run["source"]["candidate_recall"]:.5f}`. Candidate gains are accepted only if the downstream
graph also passes the predeclared precision, recall, B-cubed, false-merge, and false-split gates.

## Reproduction

```powershell
.venv\\Scripts\\shopee-entity-resolution evaluate-hybrid-candidates `
  --config configs\\experiment\\hybrid_entity_resolution.yaml
```
"""


def run_hybrid_entity_evaluation(config_path: Path) -> dict[str, object]:
    """Score frozen hybrid candidates and evaluate downstream validation clustering."""
    config = load_hybrid_entity_config(config_path)
    outputs = (
        config.artifacts.scored_pairs,
        config.artifacts.assignments,
        config.artifacts.metrics,
        config.artifacts.review,
        config.artifacts.report,
    )
    existing = [str(path) for path in outputs if path.exists()]
    if existing:
        raise OutputConflictError(
            "Refusing to overwrite hybrid-entity evidence: " + ", ".join(existing)
        )
    started = time.perf_counter()
    seed_everything(config.seed, deterministic=True)
    device = _resolve_device(config.device)
    hybrid = config.source.hybrid
    phase7 = hybrid.source.experiment
    phase6 = phase7.source.experiment
    phase5 = phase6.source.experiment
    split = load_splits(phase5.data.metadata_csv, phase5.data.split_manifest)["validation"]
    posting_ids, embeddings = _load_embeddings(hybrid.source.embedding_cache_path)
    if posting_ids != tuple(item.posting_id for item in split.items):
        raise DataValidationError("Hybrid embeddings do not align with validation")
    candidate_k = int(config.source.hybrid_metrics["selection"]["candidate_k"])
    ranking = _load_ranking(
        config.source.hybrid_ranking_path,
        posting_ids,
        expected_k=candidate_k,
    )
    model = load_phase6_model(phase7, device)
    LOGGER.info(
        "Hybrid entity stage 1/3: scoring unique pairs from hybrid Top-%d on %s",
        candidate_k,
        device,
    )
    scoring_started = time.perf_counter()
    pairs = score_candidate_pairs(
        model,
        posting_ids,
        split.items,
        embeddings,
        ranking,
        device,
        batch_size=config.pair_batch_size,
    )
    scoring_seconds = time.perf_counter() - scoring_started
    LOGGER.info("Scored %d hybrid pairs in %.2fs", len(pairs), scoring_seconds)
    LOGGER.info("Hybrid entity stage 2/3: selecting validation graph policy")
    trials, selected, assignments = sweep_recovery_policies(
        config.source.recovery,
        posting_ids,
        pairs,
        split.label_by_id,
    )
    review = _failure_review(
        split,
        assignments,
        example_limit=config.source.recovery.selection.failure_example_limit,
    )
    status = (
        "hybrid_entity_target_reached_validation_only"
        if selected["passes_quality_target"]
        else "hybrid_entity_target_not_reached_validation_only"
    )
    commit, dirty = _git_state()
    dense_incumbent = config.source.recovery.source.metrics["selection"]["selected"]["clustering"]
    run: dict[str, Any] = {
        "pipeline_version": "hybrid_entity_resolution.v1",
        "status": status,
        "provenance": {
            "config_sha256": canonical_text_sha256(config.config_path),
            "hybrid_config_sha256": config.source.hybrid_config_sha256,
            "hybrid_metrics_sha256": config.source.hybrid_metrics_sha256,
            "hybrid_ranking_sha256": config.source.hybrid_ranking_sha256,
            "recovery_config_sha256": config.source.recovery_config_sha256,
            "phase6_checkpoint_sha256": phase7.source.checkpoint_sha256,
            "split_manifest_sha256": sha256_file(phase5.data.split_manifest),
            "git_commit": commit,
            "git_dirty": dirty,
            "seed": config.seed,
            "python": platform.python_version(),
            "numpy": np.__version__,
            "device": str(device),
        },
        "data": {
            "split": "validation",
            "listings": len(posting_ids),
            "test_accessed": False,
            "model_retrained": False,
        },
        "source": {
            "candidate_k": candidate_k,
            "candidate_recall": config.source.hybrid_metrics["selection"]["recall"],
            "pair_scorer": "frozen_phase6_symmetric_pair_head",
        },
        "pair_scoring": {
            "unique_pairs": len(pairs),
            "seconds": scoring_seconds,
            "pairs_per_second": len(pairs) / scoring_seconds,
        },
        "selection": {
            "acceptance": asdict(config.source.recovery.selection.acceptance),
            "selected": selected,
            "trials": trials,
        },
        "comparison": {"dense_incumbent": dense_incumbent},
        "failure_analysis": review["counts"],
        "runtime_seconds": time.perf_counter() - started,
        "test": {"status": "disabled_hybrid_entity_validation_only"},
    }
    LOGGER.info("Hybrid entity stage 3/3: writing immutable validation evidence")
    _write_text_atomic(
        config.artifacts.scored_pairs,
        "".join(json.dumps(scored_pair_payload(pair), sort_keys=True) + "\n" for pair in pairs),
    )
    _write_assignments_atomic(config.artifacts.assignments, assignments)
    _write_text_atomic(config.artifacts.review, json.dumps(review, indent=2, sort_keys=True) + "\n")
    _write_text_atomic(config.artifacts.metrics, json.dumps(run, indent=2, sort_keys=True) + "\n")
    _write_text_atomic(config.artifacts.report, _render_report(run))
    cluster = cast(dict[str, Any], selected["clustering"])
    return {
        "status": status,
        "candidate_recall": run["source"]["candidate_recall"],
        "pairwise_precision": cluster["pairwise"]["precision"],
        "pairwise_recall": cluster["pairwise"]["recall"],
        "pairwise_f1": cluster["pairwise"]["f1"],
        "b_cubed_f1": cluster["b_cubed"]["f1"],
        "false_merge_pair_rate": cluster["false_merge_pair_rate"],
        "false_split_group_rate": cluster["false_split_group_rate"],
        "metrics": str(config.artifacts.metrics),
        "report": str(config.artifacts.report),
        "test_accessed": False,
    }
