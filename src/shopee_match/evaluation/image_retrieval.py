"""Exact cosine retrieval for normalized scratch image embeddings."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

from shopee_match.evaluation.embedding_retrieval import (
    rank_cosine_embeddings,
    rank_cosine_embeddings_profiled,
    similarity_diagnostics,
)
from shopee_match.evaluation.protocol import EvaluationSplit, Ranking

__all__ = [
    "nearest_neighbor_review",
    "rank_cosine_embeddings",
    "rank_cosine_embeddings_profiled",
    "similarity_diagnostics",
    "stratified_retrieval_metrics",
]


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
