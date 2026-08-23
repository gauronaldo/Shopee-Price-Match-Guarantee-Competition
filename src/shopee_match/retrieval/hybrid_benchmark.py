"""Validation-only hybrid dense, sparse-title, and pHash candidate retrieval."""

from __future__ import annotations

import json
import logging
import platform
import time
from pathlib import Path
from typing import Any

import numpy as np

from shopee_match.errors import DataValidationError, OutputConflictError
from shopee_match.evaluation.protocol import Ranking, load_splits, retrieval_metrics
from shopee_match.features.image import rank_phash
from shopee_match.features.text import CharTfidfModel
from shopee_match.hashing import canonical_text_sha256, sha256_file
from shopee_match.reproducibility import seed_everything
from shopee_match.retrieval.benchmark import _write_text_atomic
from shopee_match.retrieval.hybrid import reciprocal_rank_fusion
from shopee_match.retrieval.hybrid_config import load_hybrid_retrieval_config
from shopee_match.retrieval.vector_index import ExactCosineIndex, search_result_to_ranking
from shopee_match.training.multimodal_trainer import _git_state

LOGGER = logging.getLogger(__name__)


def _truncate(ranking: Ranking, top_k: int) -> Ranking:
    return {query_id: candidates[:top_k] for query_id, candidates in ranking.items()}


def _load_embeddings(path: Path) -> tuple[tuple[str, ...], np.ndarray[Any, np.dtype[np.float32]]]:
    try:
        with np.load(path, allow_pickle=False) as payload:
            posting_ids = tuple(str(value) for value in payload["posting_ids"].tolist())
            embeddings = payload["embeddings"].astype(np.float32, copy=False)
            contract = json.loads(str(payload["contract"][0]))
    except (OSError, KeyError, ValueError, json.JSONDecodeError) as exc:
        raise DataValidationError("Cannot load frozen Phase 7 embedding cache") from exc
    if (
        contract.get("version") != "phase7.listing_embeddings.v1"
        or contract.get("split") != "validation"
        or contract.get("test_accessed") is not False
        or contract.get("normalized") is not True
        or embeddings.ndim != 2
        or embeddings.shape[0] != len(posting_ids)
        or not np.isfinite(embeddings).all()
    ):
        raise DataValidationError("Frozen Phase 7 embedding contract is invalid")
    return posting_ids, embeddings


def _ranking_jsonl(ranking: Ranking) -> str:
    return "".join(
        json.dumps(
            {
                "query_posting_id": query_id,
                "candidates": [
                    {"posting_id": candidate.posting_id, "score": candidate.score}
                    for candidate in ranking[query_id]
                ],
            },
            sort_keys=True,
        )
        + "\n"
        for query_id in sorted(ranking)
    )


def _render_report(run: dict[str, Any]) -> str:
    curve_rows = "\n".join(
        f"| {candidate_k} | {metrics['recall@' + candidate_k]:.5f} | "
        f"{metrics['map@' + candidate_k]:.5f} | {metrics['precision@' + candidate_k]:.5f} |"
        for candidate_k, metrics in run["hybrid"]["retrieval_curve"].items()
    )
    selected = run["selection"]
    return f"""# Hybrid Candidate Retrieval

Status: **{run["status"]}**. Character TF-IDF statistics are fitted on train only; candidate
quality is measured on validation only and test remains untouched.

The candidate list combines frozen dense multimodal neighbours, train-fitted character TF-IDF
neighbours, and pHash neighbours with weighted reciprocal-rank fusion. Candidate IDs are
deduplicated before Top-K truncation.

| Candidate K | Recall@K | mAP@K | Precision@K |
|---:|---:|---:|---:|
{curve_rows}

- Dense Top-50 reference recall: `{run["source"]["dense_recall_at_50"]:.5f}`
- Selected hybrid K: `{selected["candidate_k"]}`
- Selected hybrid recall: `{selected["recall"]:.5f}`
- Target recall: `{selected["target_recall"]:.5f}`
- Target reached: `{str(selected["target_reached"]).lower()}`

## Reproduction

```powershell
.venv\\Scripts\\shopee-retrieval hybrid `
  --config configs\\experiment\\hybrid_candidate_retrieval.yaml
```
"""


def run_hybrid_retrieval_benchmark(config_path: Path) -> dict[str, object]:
    """Fit train-only sparse statistics and evaluate a validation candidate union."""
    config = load_hybrid_retrieval_config(config_path)
    existing = [
        str(path)
        for path in (config.artifacts.ranking, config.artifacts.metrics, config.artifacts.report)
        if path.exists()
    ]
    if existing:
        raise OutputConflictError(
            "Refusing to overwrite hybrid-retrieval evidence: " + ", ".join(existing)
        )
    started = time.perf_counter()
    seed_everything(config.seed, deterministic=True)
    phase7 = config.source.experiment
    phase6 = phase7.source.experiment
    phase5 = phase6.source.experiment
    splits = load_splits(phase5.data.metadata_csv, phase5.data.split_manifest)
    validation = splits["validation"]
    posting_ids, embeddings = _load_embeddings(config.source.embedding_cache_path)
    expected_ids = tuple(item.posting_id for item in validation.items)
    if posting_ids != expected_ids:
        raise DataValidationError("Phase 7 embeddings do not align with validation")

    LOGGER.info("Hybrid retrieval stage 1/4: exact dense Top-%d", config.fusion.dense_candidate_k)
    exact = ExactCosineIndex(posting_ids, embeddings)
    indices, scores = exact.search(
        embeddings,
        config.fusion.dense_candidate_k,
        query_ids=posting_ids,
        block_size=phase7.exact.block_size,
    )
    dense = search_result_to_ranking(posting_ids, posting_ids, indices, scores)
    LOGGER.info("Hybrid retrieval stage 2/4: fitting train-only character TF-IDF")
    tfidf_model = CharTfidfModel.fit(
        splits["train"].items,
        config.tfidf.ngram_range,
        config.tfidf.max_features,
    )
    tfidf = tfidf_model.rank(validation.items, config.tfidf.candidate_k)
    LOGGER.info("Hybrid retrieval stage 3/4: pHash ranking and reciprocal-rank fusion")
    phash = rank_phash(validation.items, config.fusion.phash_candidate_k)
    maximum_k = max(config.fusion.evaluation_k_values)
    hybrid = reciprocal_rank_fusion(
        {"dense": dense, "tfidf": tfidf, "phash": phash},
        weights={
            "dense": config.fusion.dense_weight,
            "tfidf": config.fusion.tfidf_weight,
            "phash": config.fusion.phash_weight,
        },
        rrf_constant=config.fusion.rrf_constant,
        top_k=maximum_k,
    )
    curve = {
        str(candidate_k): retrieval_metrics(
            _truncate(hybrid, candidate_k),
            validation.label_by_id,
            (candidate_k,),
            candidate_k,
        )
        for candidate_k in config.fusion.evaluation_k_values
    }
    selected_k = next(
        (
            candidate_k
            for candidate_k in config.fusion.evaluation_k_values
            if curve[str(candidate_k)][f"recall@{candidate_k}"] >= config.fusion.target_recall
        ),
        max(config.fusion.evaluation_k_values),
    )
    selected_recall = float(curve[str(selected_k)][f"recall@{selected_k}"])
    target_reached = selected_recall >= config.fusion.target_recall
    commit, dirty = _git_state()
    run: dict[str, Any] = {
        "pipeline_version": "hybrid_candidate_retrieval.v1",
        "status": (
            "hybrid_candidate_target_reached_validation_only"
            if target_reached
            else "hybrid_candidate_target_not_reached_validation_only"
        ),
        "provenance": {
            "config_sha256": canonical_text_sha256(config.config_path),
            "phase7_config_sha256": config.source.phase7_config_sha256,
            "phase7_metrics_sha256": config.source.phase7_metrics_sha256,
            "embedding_cache_sha256": config.source.embedding_cache_sha256,
            "split_manifest_sha256": sha256_file(phase5.data.split_manifest),
            "git_commit": commit,
            "git_dirty": dirty,
            "seed": config.seed,
            "python": platform.python_version(),
            "numpy": np.__version__,
        },
        "data": {
            "tfidf_fit_split": "train",
            "evaluation_split": "validation",
            "validation_listings": len(posting_ids),
            "test_accessed": False,
        },
        "source": {
            "dense_recall_at_50": config.source.metrics["exact"]["retrieval_curve"]["50"][
                "recall@50"
            ]
        },
        "fusion": {
            "method": "weighted_reciprocal_rank_fusion",
            "rrf_constant": config.fusion.rrf_constant,
            "weights": {
                "dense": config.fusion.dense_weight,
                "tfidf": config.fusion.tfidf_weight,
                "phash": config.fusion.phash_weight,
            },
        },
        "components": {
            "dense": retrieval_metrics(
                dense,
                validation.label_by_id,
                (config.fusion.dense_candidate_k,),
                config.fusion.dense_candidate_k,
            ),
            "tfidf": retrieval_metrics(
                tfidf,
                validation.label_by_id,
                (config.tfidf.candidate_k,),
                config.tfidf.candidate_k,
            ),
            "phash": retrieval_metrics(
                phash,
                validation.label_by_id,
                (config.fusion.phash_candidate_k,),
                config.fusion.phash_candidate_k,
            ),
        },
        "hybrid": {"retrieval_curve": curve},
        "selection": {
            "candidate_k": selected_k,
            "recall": selected_recall,
            "target_recall": config.fusion.target_recall,
            "target_reached": target_reached,
        },
        "runtime_seconds": time.perf_counter() - started,
        "test": {"status": "disabled_hybrid_retrieval_validation_only"},
    }
    LOGGER.info("Hybrid retrieval stage 4/4: writing frozen validation evidence")
    _write_text_atomic(config.artifacts.ranking, _ranking_jsonl(_truncate(hybrid, selected_k)))
    _write_text_atomic(config.artifacts.metrics, json.dumps(run, indent=2, sort_keys=True) + "\n")
    _write_text_atomic(config.artifacts.report, _render_report(run))
    return {
        "status": run["status"],
        "candidate_k": selected_k,
        "validation_recall": selected_recall,
        "dense_recall_at_50": run["source"]["dense_recall_at_50"],
        "metrics": str(config.artifacts.metrics),
        "ranking": str(config.artifacts.ranking),
        "report": str(config.artifacts.report),
        "test_accessed": False,
    }
