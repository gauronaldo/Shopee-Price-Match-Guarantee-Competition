from __future__ import annotations

import pytest

from shopee_match.clustering.graph import (
    ClusterAssignment,
    ScoredPair,
    build_conservative_clusters,
)
from shopee_match.clustering.metrics import clustering_metrics, edge_metrics


def _pair(
    left: int,
    right: int,
    probability: float,
    *,
    left_rank: int = 1,
    right_rank: int = 1,
    variant_conflict: bool = False,
) -> ScoredPair:
    return ScoredPair(
        left_posting_id=f"p{left}",
        right_posting_id=f"p{right}",
        left_index=left - 1,
        right_index=right - 1,
        cosine_similarity=probability,
        pair_probability=probability,
        left_rank=left_rank,
        right_rank=right_rank,
        variant_conflict=variant_conflict,
    )


def test_transitive_consistency_blocks_single_bridge_between_components() -> None:
    posting_ids = ("p1", "p2", "p3", "p4")
    pairs = [_pair(1, 2, 0.99), _pair(3, 4, 0.98), _pair(2, 3, 0.97)]
    assignments, diagnostics = build_conservative_clusters(
        posting_ids,
        pairs,
        pair_probability_threshold=0.9,
        reciprocal_rank=5,
        cross_component_minimum_coverage=1.0,
        variant_conflict_override_probability=0.95,
        maximum_cluster_size=10,
        manual_review_margin=0.02,
    )
    entities = {row.posting_id: row.entity_id for row in assignments}
    assert entities["p1"] == entities["p2"]
    assert entities["p3"] == entities["p4"]
    assert entities["p1"] != entities["p3"]
    assert diagnostics.consistency_rejections == 1


def test_reciprocal_and_variant_gates_are_label_blind() -> None:
    posting_ids = ("p1", "p2", "p3")
    pairs = [
        _pair(1, 2, 0.90, right_rank=6),
        _pair(2, 3, 0.94, variant_conflict=True),
    ]
    assignments, diagnostics = build_conservative_clusters(
        posting_ids,
        pairs,
        pair_probability_threshold=0.8,
        reciprocal_rank=5,
        cross_component_minimum_coverage=0.0,
        variant_conflict_override_probability=0.95,
        maximum_cluster_size=10,
        manual_review_margin=0.02,
    )
    assert len({row.entity_id for row in assignments}) == 3
    assert diagnostics.non_reciprocal == 1
    assert diagnostics.variant_conflict_rejected == 1


def test_clustering_metrics_match_hand_computed_perfect_partition() -> None:
    assignments = [
        ClusterAssignment("p1", "e1", 2, 0.9, False),
        ClusterAssignment("p2", "e1", 2, 0.9, False),
        ClusterAssignment("p3", "e2", 2, 0.9, False),
        ClusterAssignment("p4", "e2", 2, 0.9, False),
    ]
    labels = {"p1": "a", "p2": "a", "p3": "b", "p4": "b"}
    metrics = clustering_metrics(assignments, labels)
    assert metrics["pairwise"]["f1"] == pytest.approx(1.0)
    assert metrics["b_cubed"]["f1"] == pytest.approx(1.0)
    assert metrics["false_merge_pair_rate"] == pytest.approx(0.0)
    assert metrics["false_split_group_rate"] == pytest.approx(0.0)


def test_edge_metrics_use_all_true_pairs_as_recall_denominator() -> None:
    labels = {"p1": "a", "p2": "a", "p3": "a", "p4": "b"}
    pairs = [_pair(1, 2, 0.9), _pair(1, 4, 0.8)]
    metrics = edge_metrics(
        pairs,
        labels,
        pair_probability_threshold=0.5,
        reciprocal_rank=5,
        variant_conflict_override_probability=0.95,
    )
    assert metrics["precision"] == pytest.approx(0.5)
    assert metrics["recall"] == pytest.approx(1 / 3)
    assert metrics["false_negative"] == pytest.approx(2)
