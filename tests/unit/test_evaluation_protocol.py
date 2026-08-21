from __future__ import annotations

import json
from pathlib import Path

import pytest

from shopee_match.evaluation.protocol import (
    Ranking,
    ScoredCandidate,
    load_named_split,
    pair_metrics_at_threshold,
    precision_at_minimum_recall,
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


def test_precision_at_minimum_recall_uses_global_retrieved_pairs() -> None:
    labels = {"a": "x", "b": "x", "c": "y", "d": "y"}
    ranking = {
        "a": [ScoredCandidate("b", 0.9), ScoredCandidate("c", 0.85)],
        "b": [ScoredCandidate("a", 0.8), ScoredCandidate("c", 0.3)],
        "c": [ScoredCandidate("d", 0.7), ScoredCandidate("a", 0.4)],
        "d": [ScoredCandidate("a", 0.65), ScoredCandidate("c", 0.6)],
    }
    result = precision_at_minimum_recall(ranking, labels, minimum_recall=0.75)
    assert result["recall"] == pytest.approx(0.75)
    assert result["precision"] == pytest.approx(0.75)
    assert result["threshold"] == pytest.approx(0.7)


def test_load_named_split_does_not_retain_other_split_labels(tmp_path: Path) -> None:
    metadata = tmp_path / "train.csv"
    metadata.write_text(
        "posting_id,image,image_phash,title,label_group\n"
        "train_id,a.jpg,0000,train title,train_label\n"
        "validation_id,b.jpg,1111,validation title,validation_label\n"
        "test_id,c.jpg,2222,test title,test_label\n",
        encoding="utf-8",
    )
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        "".join(
            json.dumps({"posting_id": posting_id, "split": split}) + "\n"
            for posting_id, split in (
                ("train_id", "train"),
                ("validation_id", "validation"),
                ("test_id", "test"),
            )
        ),
        encoding="utf-8",
    )
    validation = load_named_split(metadata, manifest, "validation")
    assert tuple(item.posting_id for item in validation.items) == ("validation_id",)
    assert validation.label_by_id == {"validation_id": "validation_label"}
