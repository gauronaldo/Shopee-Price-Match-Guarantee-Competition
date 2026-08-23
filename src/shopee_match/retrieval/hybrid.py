"""Deterministic rank fusion for hybrid candidate generation."""

from __future__ import annotations

from shopee_match.evaluation.protocol import Ranking, ScoredCandidate


def reciprocal_rank_fusion(
    rankings: dict[str, Ranking],
    *,
    weights: dict[str, float],
    rrf_constant: int,
    top_k: int,
) -> Ranking:
    """Fuse label-blind rankings while deduplicating candidates by posting ID."""
    if set(rankings) != set(weights) or not rankings:
        raise ValueError("rankings and weights must contain the same non-empty source names")
    if rrf_constant <= 0 or top_k <= 0 or any(weight <= 0 for weight in weights.values()):
        raise ValueError("RRF constant, top K, and source weights must be positive")
    query_sets = [set(ranking) for ranking in rankings.values()]
    if any(query_set != query_sets[0] for query_set in query_sets[1:]):
        raise ValueError("all ranking sources must contain the same query IDs")
    result: Ranking = {}
    for query_id in sorted(query_sets[0]):
        scores: dict[str, float] = {}
        for source_name in sorted(rankings):
            weight = weights[source_name]
            for rank, candidate in enumerate(rankings[source_name][query_id], start=1):
                if candidate.posting_id == query_id:
                    continue
                scores[candidate.posting_id] = scores.get(candidate.posting_id, 0.0) + (
                    weight / (rrf_constant + rank)
                )
        ordered = sorted(scores.items(), key=lambda row: (-row[1], row[0]))[:top_k]
        result[query_id] = [ScoredCandidate(posting_id, score) for posting_id, score in ordered]
    return result
