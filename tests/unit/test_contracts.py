from pathlib import Path

import pytest

from shopee_match.contracts import CandidateMatch, OnlineQuery, OnlineResult
from shopee_match.errors import ContractError


def test_online_contract_supports_abstention() -> None:
    query = OnlineQuery(Path("query.jpg"), "Coffee 500g")
    candidate = CandidateMatch("posting_1", 0.61, 0.74, 0.52, "needs_review")
    result = OnlineResult((candidate,), None, True, True, "model.v1", "index.v1")

    assert query.title == "Coffee 500g"
    assert result.no_confident_match is True
    assert result.manual_review is True


def test_candidate_rejects_non_probability_similarity() -> None:
    with pytest.raises(ContractError, match="image_similarity"):
        CandidateMatch("posting_1", 0.7, 1.1, 0.5, "needs_review")


def test_abstention_cannot_claim_a_group() -> None:
    with pytest.raises(ContractError, match="predicted_group"):
        OnlineResult((), "group_1", True, False, "model.v1", "index.v1")
