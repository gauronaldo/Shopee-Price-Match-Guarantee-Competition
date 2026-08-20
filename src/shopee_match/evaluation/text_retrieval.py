"""Text-specific retrieval diagnostics over normalized title embeddings."""

from __future__ import annotations

from collections import Counter, defaultdict

from shopee_match.evaluation.protocol import EvaluationSplit, Ranking
from shopee_match.features.text import normalize_title


def stratified_text_retrieval_metrics(
    ranking: Ranking,
    split: EvaluationSplit,
    recall_at: tuple[int, ...],
    average_precision_at: int,
) -> dict[str, dict[str, dict[str, float]]]:
    """Report query retrieval by group size and normalized title length."""
    labels = split.label_by_id
    label_sizes = Counter(labels.values())
    items = {item.posting_id: item for item in split.items}
    members: dict[str, set[str]] = defaultdict(set)
    for posting_id, label in labels.items():
        members[label].add(posting_id)

    def group_band(size: int) -> str:
        if size <= 2:
            return "2"
        if size <= 5:
            return "3_to_5"
        if size <= 9:
            return "6_to_9"
        return "10_plus"

    def length_band(length: int) -> str:
        if length <= 30:
            return "0_to_30"
        if length <= 60:
            return "31_to_60"
        if length <= 100:
            return "61_to_100"
        return "101_plus"

    strata: dict[str, dict[str, list[str]]] = {
        "group_size": defaultdict(list),
        "normalized_title_length": defaultdict(list),
    }
    for query_id, label in labels.items():
        strata["group_size"][group_band(label_sizes[label])].append(query_id)
        title_length = len(normalize_title(items[query_id].title))
        strata["normalized_title_length"][length_band(title_length)].append(query_id)

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


def title_length_summary(split: EvaluationSplit, maximum_length: int) -> dict[str, float]:
    """Summarize normalized lengths and deterministic truncation rate."""
    lengths = sorted(len(normalize_title(item.title)) for item in split.items)
    if not lengths:
        return {"count": 0.0}

    def percentile(fraction: float) -> float:
        index = round((len(lengths) - 1) * fraction)
        return float(lengths[index])

    return {
        "count": float(len(lengths)),
        "minimum": float(lengths[0]),
        "p50": percentile(0.50),
        "p95": percentile(0.95),
        "maximum": float(lengths[-1]),
        "truncated_fraction": sum(length > maximum_length for length in lengths) / len(lengths),
    }
