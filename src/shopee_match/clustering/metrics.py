"""Pair and clustering metrics for catalog entity resolution."""

from __future__ import annotations

from collections import Counter, defaultdict
from math import comb
from typing import Any

from shopee_match.clustering.graph import ClusterAssignment, ScoredPair, eligible_pairs


def _safe_divide(numerator: int | float, denominator: int | float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _f1(precision: float, recall: float) -> float:
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def edge_metrics(
    pairs: list[ScoredPair],
    label_by_id: dict[str, str],
    *,
    pair_probability_threshold: float,
    reciprocal_rank: int,
    variant_conflict_override_probability: float,
) -> dict[str, float]:
    """Measure accepted candidate edges against every true pair in the validation corpus."""
    accepted, _ = eligible_pairs(
        pairs,
        pair_probability_threshold=pair_probability_threshold,
        reciprocal_rank=reciprocal_rank,
        variant_conflict_override_probability=variant_conflict_override_probability,
    )
    true_positive = sum(
        label_by_id[pair.left_posting_id] == label_by_id[pair.right_posting_id] for pair in accepted
    )
    false_positive = len(accepted) - true_positive
    group_sizes = Counter(label_by_id.values())
    actual_positive = sum(comb(size, 2) for size in group_sizes.values() if size > 1)
    false_negative = actual_positive - true_positive
    precision = _safe_divide(true_positive, len(accepted))
    recall = _safe_divide(true_positive, actual_positive)
    return {
        "precision": precision,
        "recall": recall,
        "f1": _f1(precision, recall),
        "true_positive": float(true_positive),
        "false_positive": float(false_positive),
        "false_negative": float(false_negative),
        "accepted_edges": float(len(accepted)),
        "actual_positive_pairs": float(actual_positive),
    }


def clustering_metrics(
    assignments: list[ClusterAssignment], label_by_id: dict[str, str]
) -> dict[str, Any]:
    """Return pairwise, B-cubed, false-merge, false-split, and size metrics."""
    entity_by_id = {row.posting_id: row.entity_id for row in assignments}
    if set(entity_by_id) != set(label_by_id):
        raise ValueError("assignments must contain every labeled posting exactly once")
    predicted: dict[str, list[str]] = defaultdict(list)
    truth: dict[str, list[str]] = defaultdict(list)
    for posting_id, entity_id in entity_by_id.items():
        predicted[entity_id].append(posting_id)
        truth[label_by_id[posting_id]].append(posting_id)

    predicted_positive = sum(comb(len(members), 2) for members in predicted.values())
    actual_positive = sum(comb(len(members), 2) for members in truth.values())
    true_positive = 0
    impure_clusters = 0
    impure_non_singletons = 0
    non_singletons = 0
    for members in predicted.values():
        label_counts = Counter(label_by_id[posting_id] for posting_id in members)
        true_positive += sum(comb(size, 2) for size in label_counts.values() if size > 1)
        impure = len(label_counts) > 1
        impure_clusters += int(impure)
        if len(members) > 1:
            non_singletons += 1
            impure_non_singletons += int(impure)
    pair_precision = _safe_divide(true_positive, predicted_positive)
    pair_recall = _safe_divide(true_positive, actual_positive)

    b3_precision_values: list[float] = []
    b3_recall_values: list[float] = []
    for posting_id in sorted(label_by_id):
        predicted_members = predicted[entity_by_id[posting_id]]
        true_members = truth[label_by_id[posting_id]]
        overlap = len(set(predicted_members) & set(true_members))
        b3_precision_values.append(overlap / len(predicted_members))
        b3_recall_values.append(overlap / len(true_members))
    b3_precision = sum(b3_precision_values) / len(b3_precision_values)
    b3_recall = sum(b3_recall_values) / len(b3_recall_values)

    split_groups = sum(
        len({entity_by_id[posting_id] for posting_id in members}) > 1 for members in truth.values()
    )
    cluster_sizes = [len(members) for members in predicted.values()]
    review_entities = {row.entity_id for row in assignments if row.manual_review}
    return {
        "pairwise": {
            "precision": pair_precision,
            "recall": pair_recall,
            "f1": _f1(pair_precision, pair_recall),
            "true_positive_pairs": float(true_positive),
            "predicted_positive_pairs": float(predicted_positive),
            "actual_positive_pairs": float(actual_positive),
        },
        "b_cubed": {
            "precision": b3_precision,
            "recall": b3_recall,
            "f1": _f1(b3_precision, b3_recall),
        },
        "false_merge_pair_rate": 1 - pair_precision if predicted_positive else 0.0,
        "impure_cluster_rate": _safe_divide(impure_clusters, len(predicted)),
        "impure_non_singleton_cluster_rate": _safe_divide(impure_non_singletons, non_singletons),
        "false_split_group_rate": _safe_divide(split_groups, len(truth)),
        "clusters": float(len(predicted)),
        "singleton_clusters": float(sum(size == 1 for size in cluster_sizes)),
        "non_singleton_clusters": float(non_singletons),
        "maximum_cluster_size": float(max(cluster_sizes, default=0)),
        "mean_cluster_size": _safe_divide(len(assignments), len(predicted)),
        "manual_review_clusters": float(len(review_entities)),
    }


def group_size_strata(
    assignments: list[ClusterAssignment], label_by_id: dict[str, str]
) -> dict[str, dict[str, float]]:
    """Report per-listing exact-entity recovery and predicted fragmentation by true group size."""
    entity_by_id = {row.posting_id: row.entity_id for row in assignments}
    truth: dict[str, list[str]] = defaultdict(list)
    for posting_id, label in label_by_id.items():
        truth[label].append(posting_id)

    def band(size: int) -> str:
        if size == 2:
            return "2"
        if size <= 5:
            return "3_to_5"
        if size <= 9:
            return "6_to_9"
        return "10_plus"

    accumulators: dict[str, dict[str, float]] = defaultdict(
        lambda: {"groups": 0.0, "listings": 0.0, "unsplit_groups": 0.0, "mean_fragments": 0.0}
    )
    for members in truth.values():
        key = band(len(members))
        fragments = len({entity_by_id[posting_id] for posting_id in members})
        row = accumulators[key]
        row["groups"] += 1
        row["listings"] += len(members)
        row["unsplit_groups"] += float(fragments == 1)
        row["mean_fragments"] += fragments
    result: dict[str, dict[str, float]] = {}
    for key in ("2", "3_to_5", "6_to_9", "10_plus"):
        row = accumulators[key]
        groups = row["groups"]
        result[key] = {
            "groups": groups,
            "listings": row["listings"],
            "unsplit_group_rate": _safe_divide(row["unsplit_groups"], groups),
            "mean_predicted_fragments": _safe_divide(row["mean_fragments"], groups),
        }
    return result
