"""Score fusion and candidate-union utilities."""

from __future__ import annotations

from shopee_match.evaluation.protocol import Ranking, ScoredCandidate


def candidate_union(rankings: tuple[Ranking, ...], per_source_k: int) -> Ranking:
    """Create a deterministic label-blind union of candidate lists."""
    queries = set(rankings[0])
    if any(set(ranking) != queries for ranking in rankings[1:]):
        raise ValueError("Rankings have different query sets")
    result: Ranking = {}
    for query in sorted(queries):
        ids = {
            candidate.posting_id
            for ranking in rankings
            for candidate in ranking[query][:per_source_k]
        }
        result[query] = [ScoredCandidate(posting_id, 0.0) for posting_id in sorted(ids)]
    return result


def fuse_rankings(
    visual: Ranking,
    text: Ranking,
    text_weight: float,
    top_k: int,
) -> Ranking:
    """Late-fuse two [0, 1] similarities; missing candidates receive zero evidence."""
    if not 0.0 <= text_weight <= 1.0:
        raise ValueError("text_weight must be in [0, 1]")
    if set(visual) != set(text):
        raise ValueError("Rankings have different query sets")
    result: Ranking = {}
    for query_id in sorted(visual):
        visual_scores = {item.posting_id: item.score for item in visual[query_id]}
        text_scores = {item.posting_id: item.score for item in text[query_id]}
        candidate_ids = visual_scores.keys() | text_scores.keys()
        candidates = [
            ScoredCandidate(
                posting_id,
                (1 - text_weight) * visual_scores.get(posting_id, 0.0)
                + text_weight * text_scores.get(posting_id, 0.0),
            )
            for posting_id in candidate_ids
        ]
        result[query_id] = sorted(
            candidates, key=lambda candidate: (-candidate.score, candidate.posting_id)
        )[:top_k]
    return result
