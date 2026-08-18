from __future__ import annotations

import pytest

from shopee_match.evaluation.protocol import (
    Ranking,
    ScoredCandidate,
    pair_metrics_at_threshold,
    retrieval_metrics,
    select_threshold,
)


def hand_checkable_ranking() -> tuple[Ranking, dict[str, str]]:
    labels = {"a": "g1", "b": "g1", "c": "g2", "d": "g2"}
    ranking = {
        "a": [ScoredCandidate("b", 0.9), ScoredCandidate("c", 0.2)],
        "b": [ScoredCandidate("a", 0.8), ScoredCandidate("c", 0.3)],
        "c": [ScoredCandidate("a", 0.7), ScoredCandidate("d", 0.6)],
        "d": [ScoredCandidate("c", 0.95), ScoredCandidate("a", 0.1)],
    }
    return ranking, labels


def test_retrieval_metrics_match_hand_computation() -> None:
    ranking, labels = hand_checkable_ranking()

    metrics = retrieval_metrics(ranking, labels, (1,), 2)

    assert metrics["hit_rate@1"] == pytest.approx(0.75)
    assert metrics["precision@1"] == pytest.approx(0.75)
    assert metrics["recall@1"] == pytest.approx(0.75)
    assert metrics["f1@1"] == pytest.approx(0.75)
    assert metrics["map@2"] == pytest.approx(0.875)


def test_threshold_selection_counts_unretrieved_positives() -> None:
    ranking, labels = hand_checkable_ranking()

    selected = select_threshold(ranking, labels)
    applied = pair_metrics_at_threshold(ranking, labels, selected["threshold"])

    assert selected == applied
    assert selected["threshold"] == pytest.approx(0.6)
    assert selected["true_positive"] == 4
    assert selected["false_positive"] == 1
    assert selected["false_negative"] == 0


def test_metrics_reject_non_deterministic_ranking() -> None:
    ranking, labels = hand_checkable_ranking()
    ranking["a"].reverse()

    with pytest.raises(ValueError, match="deterministically ranked"):
        retrieval_metrics(ranking, labels, (1,), 2)
