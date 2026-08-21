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


def candidate_pair_classification_metrics(
    pairs: list[ScoredPair],
    label_by_id: dict[str, str],
    *,
    threshold: float,
    calibration_bins: int,
    required_recall: float,
    required_precision: float,
) -> dict[str, float]:
    """Evaluate raw pair-head probabilities within the frozen retrieved candidate set."""
    if not pairs:
        raise ValueError("pairs must be non-empty")
    if calibration_bins <= 0:
        raise ValueError("calibration_bins must be positive")
    if not all(0 <= value <= 1 for value in (threshold, required_recall, required_precision)):
        raise ValueError("threshold and operating requirements must be inside [0, 1]")

    scored: list[tuple[float, bool, str, str]] = []
    for pair in pairs:
        if not 0 <= pair.pair_probability <= 1:
            raise ValueError("pair probabilities must be inside [0, 1]")
        try:
            positive = label_by_id[pair.left_posting_id] == label_by_id[pair.right_posting_id]
        except KeyError as exc:
            raise ValueError("pair references an unknown posting ID") from exc
        scored.append(
            (
                pair.pair_probability,
                positive,
                pair.left_posting_id,
                pair.right_posting_id,
            )
        )
    scored.sort(key=lambda row: (-row[0], row[2], row[3]))
    positive_count = sum(row[1] for row in scored)
    negative_count = len(scored) - positive_count
    if positive_count == 0 or negative_count == 0:
        raise ValueError("candidate pairs must contain both positive and negative examples")

    true_positive = sum(row[1] and row[0] >= threshold for row in scored)
    false_positive = sum(not row[1] and row[0] >= threshold for row in scored)
    false_negative = positive_count - true_positive
    true_negative = negative_count - false_positive
    precision = _safe_divide(true_positive, true_positive + false_positive)
    recall = _safe_divide(true_positive, positive_count)

    cumulative_positive = 0
    average_precision_sum = 0.0
    precision_at_required_recall = 0.0
    recall_at_required_precision = 0.0
    for rank, (_score, positive, _left, _right) in enumerate(scored, start=1):
        cumulative_positive += int(positive)
        prefix_precision = cumulative_positive / rank
        prefix_recall = cumulative_positive / positive_count
        if positive:
            average_precision_sum += prefix_precision
        if prefix_recall >= required_recall:
            precision_at_required_recall = max(precision_at_required_recall, prefix_precision)
        if prefix_precision >= required_precision:
            recall_at_required_precision = max(recall_at_required_precision, prefix_recall)

    brier = sum((score - float(positive)) ** 2 for score, positive, _left, _right in scored) / len(
        scored
    )
    calibration_error = 0.0
    for bin_index in range(calibration_bins):
        lower = bin_index / calibration_bins
        upper = (bin_index + 1) / calibration_bins
        members = [
            row
            for row in scored
            if row[0] >= lower
            and (row[0] < upper or (bin_index == calibration_bins - 1 and row[0] <= upper))
        ]
        if not members:
            continue
        mean_probability = sum(row[0] for row in members) / len(members)
        observed_rate = sum(row[1] for row in members) / len(members)
        calibration_error += len(members) / len(scored) * abs(mean_probability - observed_rate)

    return {
        "threshold": threshold,
        "precision": precision,
        "recall_within_candidates": recall,
        "f1_within_candidates": _f1(precision, recall),
        "average_precision_pr_auc": average_precision_sum / positive_count,
        "brier_score": brier,
        "expected_calibration_error": calibration_error,
        "precision_at_required_recall": precision_at_required_recall,
        "required_recall": required_recall,
        "recall_at_required_precision": recall_at_required_precision,
        "required_precision": required_precision,
        "true_positive": float(true_positive),
        "false_positive": float(false_positive),
        "true_negative": float(true_negative),
        "false_negative_within_candidates": float(false_negative),
        "candidate_positive_pairs": float(positive_count),
        "candidate_negative_pairs": float(negative_count),
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
