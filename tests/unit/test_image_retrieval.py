from __future__ import annotations

import numpy as np

from shopee_match.evaluation.image_retrieval import (
    nearest_neighbor_review,
    rank_cosine_embeddings,
    rank_cosine_embeddings_profiled,
    similarity_diagnostics,
    stratified_retrieval_metrics,
)
from shopee_match.evaluation.protocol import CorpusItem, EvaluationSplit, retrieval_metrics


def test_exact_cosine_retrieval_finds_hand_checkable_product_pairs() -> None:
    posting_ids = ("a1", "a2", "b1", "b2")
    embeddings = np.asarray([[1.0, 0.0], [0.99, 0.01], [0.0, 1.0], [0.01, 0.99]], dtype=np.float32)
    labels = {"a1": "a", "a2": "a", "b1": "b", "b2": "b"}

    ranking = rank_cosine_embeddings(posting_ids, embeddings, candidate_k=3)
    metrics = retrieval_metrics(ranking, labels, recall_at=(1, 3), average_precision_at=3)

    assert ranking["a1"][0].posting_id == "a2"
    assert ranking["b1"][0].posting_id == "b2"
    assert metrics["recall@1"] == 1.0
    assert metrics["map@3"] == 1.0


def test_image_retrieval_diagnostics_are_bounded_and_stratified() -> None:
    posting_ids = ("a1", "a2", "b1", "b2")
    embeddings = np.asarray([[1.0, 0.0], [0.99, 0.01], [0.0, 1.0], [0.01, 0.99]], dtype=np.float32)
    labels = {"a1": "a", "a2": "a", "b1": "b", "b2": "b"}
    items = tuple(
        CorpusItem(posting_id, f"{posting_id}.png", label * 16, posting_id)
        for posting_id, label in zip(posting_ids, ("a", "a", "b", "b"), strict=True)
    )
    split = EvaluationSplit(items, labels)

    ranking, latency = rank_cosine_embeddings_profiled(posting_ids, embeddings, 3)
    distributions = similarity_diagnostics(
        posting_ids, embeddings, labels, seed=3, maximum_pairs_per_class=10
    )
    strata = stratified_retrieval_metrics(ranking, split, (1, 3), 3)
    review = nearest_neighbor_review(ranking, split, limit_per_bucket=2)

    assert latency["ranking_p95_ms_per_query"] >= 0
    assert distributions["positive"]["mean"] > distributions["negative"]["mean"]
    assert strata["group_size"]["2"]["map@3"] == 1.0
    assert len(review["top1_success"]) == 2
    assert not review["top1_false_match"]
