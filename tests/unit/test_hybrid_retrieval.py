from __future__ import annotations

import pytest

from shopee_match.evaluation.protocol import ScoredCandidate
from shopee_match.retrieval.hybrid import reciprocal_rank_fusion


def test_reciprocal_rank_fusion_deduplicates_and_is_deterministic() -> None:
    dense = {
        "q": [ScoredCandidate("a", 0.9), ScoredCandidate("b", 0.8)],
    }
    sparse = {
        "q": [ScoredCandidate("b", 0.7), ScoredCandidate("c", 0.6)],
    }
    result = reciprocal_rank_fusion(
        {"dense": dense, "sparse": sparse},
        weights={"dense": 1.0, "sparse": 1.0},
        rrf_constant=60,
        top_k=3,
    )
    assert [row.posting_id for row in result["q"]] == ["b", "a", "c"]
    assert result["q"][0].score == pytest.approx(1 / 61 + 1 / 62)


def test_reciprocal_rank_fusion_rejects_misaligned_queries() -> None:
    with pytest.raises(ValueError, match="same query IDs"):
        reciprocal_rank_fusion(
            {
                "dense": {"q1": [ScoredCandidate("a", 1.0)]},
                "sparse": {"q2": [ScoredCandidate("a", 1.0)]},
            },
            weights={"dense": 1.0, "sparse": 1.0},
            rrf_constant=60,
            top_k=1,
        )
