"""Typed system boundaries for future retrieval, pair scoring, and clustering stages."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from shopee_match.errors import ContractError

Decision = Literal["match", "no_match", "needs_review"]


def _probability(value: float, name: str) -> None:
    if not 0 <= value <= 1:
        raise ContractError(f"{name} must be in [0, 1], got {value}")


@dataclass(frozen=True, slots=True)
class OnlineQuery:
    """One listing submitted for duplicate retrieval."""

    image_path: Path
    title: str

    def __post_init__(self) -> None:
        if not self.title.strip():
            raise ContractError("OnlineQuery.title must not be empty")


@dataclass(frozen=True, slots=True)
class CandidateMatch:
    """A ranked candidate with separate multimodal evidence."""

    posting_id: str
    match_confidence: float
    image_similarity: float
    title_similarity: float
    decision: Decision

    def __post_init__(self) -> None:
        if not self.posting_id:
            raise ContractError("CandidateMatch.posting_id must not be empty")
        _probability(self.match_confidence, "match_confidence")
        _probability(self.image_similarity, "image_similarity")
        _probability(self.title_similarity, "title_similarity")


@dataclass(frozen=True, slots=True)
class OnlineResult:
    """Versioned online response supporting abstention and review."""

    candidates: tuple[CandidateMatch, ...]
    predicted_group: str | None
    no_confident_match: bool
    manual_review: bool
    model_version: str
    index_version: str

    def __post_init__(self) -> None:
        if self.no_confident_match and self.predicted_group is not None:
            raise ContractError("no_confident_match result cannot contain predicted_group")


@dataclass(frozen=True, slots=True)
class CandidatePair:
    """Canonical unordered listing pair for batch scoring."""

    left_posting_id: str
    right_posting_id: str
    match_probability: float

    def __post_init__(self) -> None:
        if not self.left_posting_id or not self.right_posting_id:
            raise ContractError("CandidatePair posting IDs must not be empty")
        if self.left_posting_id >= self.right_posting_id:
            raise ContractError("CandidatePair IDs must be unique and lexicographically ordered")
        _probability(self.match_probability, "match_probability")


@dataclass(frozen=True, slots=True)
class ProductCluster:
    """Conservative entity-resolution output with review evidence."""

    entity_id: str
    member_posting_ids: tuple[str, ...]
    confidence: float
    review_reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.entity_id or not self.member_posting_ids:
            raise ContractError("ProductCluster requires an entity ID and at least one member")
        if len(self.member_posting_ids) != len(set(self.member_posting_ids)):
            raise ContractError("ProductCluster members must be unique")
        _probability(self.confidence, "confidence")
