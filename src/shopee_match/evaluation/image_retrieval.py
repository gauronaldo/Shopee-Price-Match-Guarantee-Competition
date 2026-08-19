"""Exact cosine retrieval for normalized scratch image embeddings."""

from __future__ import annotations

import time
from collections import Counter, defaultdict
from typing import Any

import numpy as np

from shopee_match.evaluation.protocol import EvaluationSplit, Ranking, ScoredCandidate


def _normalized(embeddings: np.ndarray) -> np.ndarray:
    if not np.isfinite(embeddings).all():
        raise ValueError("embeddings contain non-finite values")
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    if np.any(norms <= 0):
        raise ValueError("embeddings must have non-zero norm")
    return embeddings / norms


def rank_cosine_embeddings(
    posting_ids: tuple[str, ...], embeddings: np.ndarray, candidate_k: int
) -> Ranking:
    """Rank every query against its full split with deterministic ID tie-breaking."""
    ranking, _latency = rank_cosine_embeddings_profiled(posting_ids, embeddings, candidate_k)
    return ranking


def rank_cosine_embeddings_profiled(
    posting_ids: tuple[str, ...], embeddings: np.ndarray, candidate_k: int
) -> tuple[Ranking, dict[str, float]]:
    """Exact ranking plus per-query CPU/GPU-independent sorting latency percentiles."""
    if embeddings.ndim != 2 or embeddings.shape[0] != len(posting_ids):
        raise ValueError("embeddings must have shape [len(posting_ids), dimension]")
    if len(set(posting_ids)) != len(posting_ids):
        raise ValueError("posting_ids must be unique")
    if candidate_k <= 0:
        raise ValueError("candidate_k must be positive")
    normalized = _normalized(embeddings)
    matrix_started = time.perf_counter()
    similarities = normalized @ normalized.T
    matrix_seconds = time.perf_counter() - matrix_started
    id_array = np.asarray(posting_ids, dtype=str)
    limit = min(candidate_k, max(0, len(posting_ids) - 1))
    ranking: Ranking = {}
    query_latencies_ms: list[float] = []
    for query_index, query_id in enumerate(posting_ids):
        query_started = time.perf_counter()
        scores = similarities[query_index].copy()
        scores[query_index] = -np.inf
        order = np.lexsort((id_array, -scores))
        candidates = [index for index in order if index != query_index][:limit]
        ranking[query_id] = [
            ScoredCandidate(posting_ids[index], float(scores[index])) for index in candidates
        ]
        query_latencies_ms.append((time.perf_counter() - query_started) * 1000)
    latency = np.asarray(query_latencies_ms, dtype=np.float64)
    return ranking, {
        "similarity_matrix_seconds": matrix_seconds,
        "ranking_p50_ms_per_query": float(np.percentile(latency, 50)) if len(latency) else 0.0,
        "ranking_p95_ms_per_query": float(np.percentile(latency, 95)) if len(latency) else 0.0,
        "ranking_mean_ms_per_query": float(latency.mean()) if len(latency) else 0.0,
    }


def similarity_diagnostics(
    posting_ids: tuple[str, ...],
    embeddings: np.ndarray,
    label_by_id: dict[str, str],
    *,
    seed: int,
    maximum_pairs_per_class: int = 100_000,
) -> dict[str, dict[str, float]]:
    """Summarize bounded positive and negative cosine distributions."""
    if maximum_pairs_per_class <= 0:
        raise ValueError("maximum_pairs_per_class must be positive")
    normalized = _normalized(embeddings)
    indices_by_label: dict[str, list[int]] = defaultdict(list)
    for index, posting_id in enumerate(posting_ids):
        indices_by_label[label_by_id[posting_id]].append(index)
    positive_pairs = [
        (indices[left], indices[right])
        for indices in indices_by_label.values()
        for left in range(len(indices))
        for right in range(left + 1, len(indices))
    ]
    rng = np.random.default_rng(seed)
    if len(positive_pairs) > maximum_pairs_per_class:
        chosen = rng.choice(len(positive_pairs), maximum_pairs_per_class, replace=False)
        positive_pairs = [positive_pairs[int(index)] for index in chosen]

    negative_pairs: list[tuple[int, int]] = []
    attempts = 0
    maximum_attempts = maximum_pairs_per_class * 20
    while len(negative_pairs) < maximum_pairs_per_class and attempts < maximum_attempts:
        left, right = rng.integers(0, len(posting_ids), size=2)
        attempts += 1
        if left != right and label_by_id[posting_ids[left]] != label_by_id[posting_ids[right]]:
            negative_pairs.append((int(left), int(right)))

    def summarize(pairs: list[tuple[int, int]]) -> dict[str, float]:
        values = np.asarray(
            [float(normalized[left] @ normalized[right]) for left, right in pairs],
            dtype=np.float64,
        )
        if not len(values):
            return {"count": 0.0}
        return {
            "count": float(len(values)),
            "mean": float(values.mean()),
            "std": float(values.std()),
            "p05": float(np.percentile(values, 5)),
            "p50": float(np.percentile(values, 50)),
            "p95": float(np.percentile(values, 95)),
        }

    return {"positive": summarize(positive_pairs), "negative": summarize(negative_pairs)}


def stratified_retrieval_metrics(
    ranking: Ranking,
    split: EvaluationSplit,
    recall_at: tuple[int, ...],
    average_precision_at: int,
) -> dict[str, dict[str, dict[str, float]]]:
    """Compute query-level retrieval metrics by group size and exact-positive-pHash evidence."""
    labels = split.label_by_id
    label_sizes = Counter(labels.values())
    items = {item.posting_id: item for item in split.items}
    members: dict[str, set[str]] = defaultdict(set)
    for posting_id, label in labels.items():
        members[label].add(posting_id)

    def band(size: int) -> str:
        if size <= 2:
            return "2"
        if size <= 5:
            return "3_to_5"
        if size <= 9:
            return "6_to_9"
        return "10_plus"

    strata: dict[str, dict[str, list[str]]] = {
        "group_size": defaultdict(list),
        "positive_phash": defaultdict(list),
    }
    for query_id, label in labels.items():
        strata["group_size"][band(label_sizes[label])].append(query_id)
        positives = members[label] - {query_id}
        exact = any(
            items[candidate].image_phash == items[query_id].image_phash for candidate in positives
        )
        strata["positive_phash"]["has_exact_positive" if exact else "no_exact_positive"].append(
            query_id
        )

    def metrics_for(query_ids: list[str]) -> dict[str, float]:
        totals = {f"recall@{k}": 0.0 for k in recall_at}
        mean_ap = 0.0
        for query_id in query_ids:
            positives = members[labels[query_id]] - {query_id}
            candidate_ids = [candidate.posting_id for candidate in ranking[query_id]]
            for k in recall_at:
                totals[f"recall@{k}"] += len(positives.intersection(candidate_ids[:k])) / len(
                    positives
                )
            hits = 0
            precision_sum = 0.0
            for rank, candidate_id in enumerate(candidate_ids[:average_precision_at], start=1):
                if candidate_id in positives:
                    hits += 1
                    precision_sum += hits / rank
            mean_ap += precision_sum / min(len(positives), average_precision_at)
        count = len(query_ids)
        return {
            **{name: value / count for name, value in totals.items()},
            f"map@{average_precision_at}": mean_ap / count,
            "queries": float(count),
        }

    return {
        dimension: {name: metrics_for(query_ids) for name, query_ids in values.items()}
        for dimension, values in strata.items()
    }


def nearest_neighbor_review(
    ranking: Ranking, split: EvaluationSplit, *, limit_per_bucket: int = 20
) -> dict[str, list[dict[str, Any]]]:
    """Create a lightweight local review manifest without copying competition images."""
    if limit_per_bucket <= 0:
        raise ValueError("limit_per_bucket must be positive")
    labels = split.label_by_id
    items = {item.posting_id: item for item in split.items}
    result: dict[str, list[dict[str, Any]]] = {
        "top1_success": [],
        "top1_false_match": [],
        "retrieval_miss": [],
    }
    for query_id in sorted(ranking):
        candidates = ranking[query_id]
        if not candidates:
            continue
        positive_ids = {item for item, label in labels.items() if label == labels[query_id]} - {
            query_id
        }
        top = candidates[0]
        top_same = top.posting_id in positive_ids
        retrieved_positive = any(candidate.posting_id in positive_ids for candidate in candidates)
        buckets = ["top1_success" if top_same else "top1_false_match"]
        if not retrieved_positive:
            buckets.append("retrieval_miss")
        for bucket in buckets:
            if len(result[bucket]) >= limit_per_bucket:
                continue
            result[bucket].append(
                {
                    "query_id": query_id,
                    "query_image": items[query_id].image,
                    "query_label": labels[query_id],
                    "candidate_id": top.posting_id,
                    "candidate_image": items[top.posting_id].image,
                    "candidate_label": labels[top.posting_id],
                    "cosine_similarity": top.score,
                    "exact_phash": (
                        items[query_id].image_phash == items[top.posting_id].image_phash
                    ),
                    "manual_category": None,
                }
            )
    return result
