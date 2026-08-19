"""Deterministic image baselines: pHash retrieval and ORB candidate reranking."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import cv2
import numpy as np
import numpy.typing as npt

from shopee_match.evaluation.protocol import CorpusItem, Ranking, ScoredCandidate

_BIT_COUNTS = np.asarray([value.bit_count() for value in range(256)], dtype=np.uint8)


def phash_distance(left: str, right: str) -> int:
    return (int(left, 16) ^ int(right, 16)).bit_count()


def rank_phash(items: tuple[CorpusItem, ...], top_k: int) -> Ranking:
    """Rank every query against the full split by 64-bit pHash Hamming distance."""
    return rank_phash_queries(items, items, top_k)


def rank_phash_queries(
    items: tuple[CorpusItem, ...], queries: tuple[CorpusItem, ...], top_k: int
) -> Ranking:
    """Rank a bounded query subset against a full corpus by pHash distance."""
    ordered_items = sorted(items, key=lambda item: item.posting_id)
    ids = [item.posting_id for item in ordered_items]
    hashes = np.asarray([int(item.image_phash, 16) for item in ordered_items], dtype=np.uint64)
    ranking: Ranking = {}
    for query in sorted(queries, key=lambda item: item.posting_id):
        query_id = query.posting_id
        xor = np.bitwise_xor(hashes, np.uint64(int(query.image_phash, 16)))
        distances = _BIT_COUNTS[xor.view(np.uint8).reshape(-1, 8)].sum(axis=1)
        candidate_indices = sorted(
            (candidate for candidate in range(len(ids)) if ids[candidate] != query_id),
            key=lambda candidate: (int(distances[candidate]), ids[candidate]),
        )[:top_k]
        ranking[query_id] = [
            ScoredCandidate(ids[candidate], 1.0 - int(distances[candidate]) / 64.0)
            for candidate in candidate_indices
        ]
    return ranking


def _orb_descriptors(path: Path, features: int) -> npt.NDArray[np.uint8] | None:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Cannot decode image: {path}")
    orb_factory = cast(Any, cv2).ORB_create
    detector = orb_factory(nfeatures=features)
    _keypoints, descriptors = detector.detectAndCompute(image, None)
    return cast(npt.NDArray[np.uint8] | None, descriptors)


def _ratio_score(
    query: npt.NDArray[np.uint8] | None,
    candidate: npt.NDArray[np.uint8] | None,
) -> float:
    if query is None or candidate is None or len(query) < 2 or len(candidate) < 2:
        return 0.0
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = matcher.knnMatch(query, candidate, k=2)
    accepted = sum(
        1 for pair in matches if len(pair) == 2 and pair[0].distance < 0.75 * pair[1].distance
    )
    return accepted / len(matches) if matches else 0.0


def rerank_orb(
    items: tuple[CorpusItem, ...],
    candidate_ranking: Ranking,
    image_dir: Path,
    features: int,
    top_k: int,
) -> Ranking:
    """Rerank a label-blind candidate union using symmetric ORB ratio-test coverage."""
    pair_scores = score_orb_candidates(items, candidate_ranking, image_dir, features)
    return rank_orb_scores(candidate_ranking, pair_scores, top_k)


def rank_orb_scores(
    candidate_ranking: Ranking,
    pair_scores: dict[str, dict[str, float]],
    top_k: int,
) -> Ranking:
    """Rank a previously scored ORB candidate set."""
    result: Ranking = {}
    for query_id, candidates in candidate_ranking.items():
        scored = [
            ScoredCandidate(candidate.posting_id, pair_scores[query_id][candidate.posting_id])
            for candidate in candidates
        ]
        result[query_id] = sorted(
            scored, key=lambda candidate: (-candidate.score, candidate.posting_id)
        )[:top_k]
    return result


def score_orb_candidates(
    items: tuple[CorpusItem, ...],
    candidate_ranking: Ranking,
    image_dir: Path,
    features: int,
) -> dict[str, dict[str, float]]:
    """Score every supplied candidate pair without truncating the candidate set."""
    by_id = {item.posting_id: item for item in items}
    required_ids = set(candidate_ranking)
    required_ids.update(
        candidate.posting_id
        for candidates in candidate_ranking.values()
        for candidate in candidates
    )
    descriptors = {
        posting_id: _orb_descriptors(image_dir / by_id[posting_id].image, features)
        for posting_id in sorted(required_ids)
    }
    result: dict[str, dict[str, float]] = {}
    score_cache: dict[tuple[str, str], float] = {}
    for query_id, candidates in candidate_ranking.items():
        scored: dict[str, float] = {}
        for candidate in candidates:
            pair = (
                (query_id, candidate.posting_id)
                if query_id < candidate.posting_id
                else (candidate.posting_id, query_id)
            )
            if pair not in score_cache:
                forward = _ratio_score(descriptors[query_id], descriptors[candidate.posting_id])
                backward = _ratio_score(descriptors[candidate.posting_id], descriptors[query_id])
                score_cache[pair] = (forward + backward) / 2
            scored[candidate.posting_id] = score_cache[pair]
        result[query_id] = scored
    return result
