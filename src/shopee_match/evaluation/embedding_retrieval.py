"""Modality-independent exact cosine retrieval and embedding diagnostics."""

from __future__ import annotations

import time
from collections import defaultdict
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from shopee_match.evaluation.protocol import Ranking, ScoredCandidate

FloatArray = NDArray[np.floating[Any]]


def _normalized(embeddings: FloatArray) -> FloatArray:
    if not np.isfinite(embeddings).all():
        raise ValueError("embeddings contain non-finite values")
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    if np.any(norms <= 0):
        raise ValueError("embeddings must have non-zero norm")
    return cast(FloatArray, embeddings / norms)


def rank_cosine_embeddings(
    posting_ids: tuple[str, ...], embeddings: FloatArray, candidate_k: int
) -> Ranking:
    """Rank every query against its full split with deterministic ID tie-breaking."""
    ranking, _latency = rank_cosine_embeddings_profiled(posting_ids, embeddings, candidate_k)
    return ranking


def rank_cosine_embeddings_profiled(
    posting_ids: tuple[str, ...], embeddings: FloatArray, candidate_k: int
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
    embeddings: FloatArray,
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
