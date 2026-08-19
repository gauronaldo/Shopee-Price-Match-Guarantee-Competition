"""Candidate scoring and diagnostics for the classical pair matcher."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from shopee_match.evaluation.protocol import (
    CorpusItem,
    EvaluationSplit,
    Ranking,
    ScoredCandidate,
)
from shopee_match.features.image import phash_distance
from shopee_match.features.pair import (
    FEATURE_NAMES,
    ClassicalPairModel,
    pair_feature_values,
)
from shopee_match.features.text import CharTfidfModel


@dataclass(frozen=True, slots=True)
class PairFeatureBatch:
    """Stable pair ordering plus its dense handcrafted feature matrix."""

    pairs: tuple[tuple[str, str], ...]
    values: npt.NDArray[np.float64]


def add_training_positives(candidates: Ranking, labels: dict[str, str]) -> Ranking:
    """Add all known train positives while keeping candidate generation label-blind elsewhere."""
    members: dict[str, list[str]] = defaultdict(list)
    for posting_id, label in labels.items():
        members[label].append(posting_id)
    result: Ranking = {}
    for query_id, scored in candidates.items():
        candidate_ids = {candidate.posting_id for candidate in scored}
        candidate_ids.update(member for member in members[labels[query_id]] if member != query_id)
        result[query_id] = [
            ScoredCandidate(posting_id, 0.0) for posting_id in sorted(candidate_ids)
        ]
    return result


def build_pair_features(
    items: tuple[CorpusItem, ...],
    candidates: Ranking,
    text_model: CharTfidfModel,
    orb_scores: dict[str, dict[str, float]],
) -> PairFeatureBatch:
    """Build label-blind features for an explicit candidate set."""
    by_id = {item.posting_id: item for item in items}
    required_ids = set(candidates)
    required_ids.update(
        candidate.posting_id for scored in candidates.values() for candidate in scored
    )
    text_vectors = {
        posting_id: text_model.transform_one(by_id[posting_id].title)
        for posting_id in sorted(required_ids)
    }

    def text_similarity(left_id: str, right_id: str) -> float:
        left_vector = text_vectors[left_id]
        right_vector = text_vectors[right_id]
        if len(left_vector) > len(right_vector):
            left_vector, right_vector = right_vector, left_vector
        return sum(value * right_vector.get(index, 0.0) for index, value in left_vector.items())

    pairs: list[tuple[str, str]] = []
    rows: list[npt.NDArray[np.float64]] = []
    for query_id in sorted(candidates):
        query = by_id[query_id]
        for candidate in candidates[query_id]:
            candidate_item = by_id[candidate.posting_id]
            pairs.append((query_id, candidate.posting_id))
            rows.append(
                pair_feature_values(
                    query,
                    candidate_item,
                    text_similarity(query_id, candidate.posting_id),
                    1.0 - phash_distance(query.image_phash, candidate_item.image_phash) / 64.0,
                    orb_scores[query_id][candidate.posting_id],
                )
            )
    values = np.vstack(rows) if rows else np.empty((0, len(FEATURE_NAMES)), dtype=np.float64)
    return PairFeatureBatch(tuple(pairs), values)


def training_labels(batch: PairFeatureBatch, label_by_id: dict[str, str]) -> npt.NDArray[np.int64]:
    return np.asarray(
        [int(label_by_id[left] == label_by_id[right]) for left, right in batch.pairs],
        dtype=np.int64,
    )


def rank_pair_candidates(
    candidates: Ranking,
    batch: PairFeatureBatch,
    model: ClassicalPairModel,
    top_k: int,
) -> Ranking:
    """Predict calibrated pair probabilities and return a deterministic ranking."""
    scores = model.predict_scores(batch.values)
    by_query: dict[str, list[ScoredCandidate]] = {query_id: [] for query_id in candidates}
    for (query_id, candidate_id), score in zip(batch.pairs, scores, strict=True):
        by_query[query_id].append(ScoredCandidate(candidate_id, float(score)))
    return {
        query_id: sorted(scored, key=lambda candidate: (-candidate.score, candidate.posting_id))[
            :top_k
        ]
        for query_id, scored in by_query.items()
    }


def candidate_ceiling(candidates: Ranking, labels: dict[str, str]) -> dict[str, float]:
    """Measure the maximum recall possible before candidate scoring."""
    members: dict[str, set[str]] = defaultdict(set)
    for posting_id, label in labels.items():
        members[label].add(posting_id)
    recalls: list[float] = []
    hits: list[float] = []
    counts: list[int] = []
    for query_id, scored in candidates.items():
        positives = members[labels[query_id]] - {query_id}
        candidate_ids = {candidate.posting_id for candidate in scored}
        found = len(positives & candidate_ids)
        recalls.append(found / len(positives))
        hits.append(float(found > 0))
        counts.append(len(candidate_ids))
    return {
        "macro_recall": float(np.mean(recalls)),
        "hit_rate": float(np.mean(hits)),
        "mean_candidates": float(np.mean(counts)),
        "max_candidates": float(max(counts, default=0)),
    }


def _failure_category(features: npt.NDArray[np.float64]) -> str:
    values = dict(zip(FEATURE_NAMES, features, strict=True))
    if values["quantity_conflict"]:
        return "quantity_or_unit_conflict"
    if values["digit_conflict"]:
        return "digit_or_model_conflict"
    if values["exact_normalized_title"]:
        return "exact_title_cross_label"
    if values["phash_similarity"] == 1.0:
        return "exact_phash_cross_label"
    if values["tfidf_similarity"] >= 0.7 and values["phash_similarity"] < 0.7:
        return "text_dominant_modality_disagreement"
    if values["phash_similarity"] >= 0.9 and values["tfidf_similarity"] < 0.3:
        return "image_dominant_modality_disagreement"
    return "other_pair_error"


def failure_analysis(
    batch: PairFeatureBatch,
    ranking: Ranking,
    evaluation_split: EvaluationSplit,
    threshold: float,
    sample_limit: int = 5,
) -> dict[str, Any]:
    """Count structured pair errors and retain bounded local examples."""
    labels = evaluation_split.label_by_id
    by_id = {item.posting_id: item for item in evaluation_split.items}
    feature_by_pair = dict(zip(batch.pairs, batch.values, strict=True))
    score_by_pair = {
        (query_id, candidate.posting_id): candidate.score
        for query_id, candidates in ranking.items()
        for candidate in candidates
    }
    counts: Counter[str] = Counter()
    examples: dict[str, list[dict[str, Any]]] = defaultdict(list)
    positive_pairs = {
        (left, right)
        for left in labels
        for right in labels
        if left != right and labels[left] == labels[right]
    }
    candidate_pairs = set(batch.pairs)
    ranked_pairs = set(score_by_pair)
    counts["candidate_generation_miss"] = len(positive_pairs - candidate_pairs)
    counts["scorer_top_k_miss"] = len((positive_pairs & candidate_pairs) - ranked_pairs)
    for pair, score in score_by_pair.items():
        truth = labels[pair[0]] == labels[pair[1]]
        predicted = score >= threshold
        if truth == predicted:
            continue
        category = _failure_category(feature_by_pair[pair])
        error_type = "false_positive" if predicted else "false_negative"
        key = f"{error_type}:{category}"
        counts[key] += 1
        if len(examples[key]) < sample_limit:
            left, right = pair
            examples[key].append(
                {
                    "query_id": left,
                    "query_title": by_id[left].title,
                    "candidate_id": right,
                    "candidate_title": by_id[right].title,
                    "score": score,
                    "features": {
                        name: float(value)
                        for name, value in zip(FEATURE_NAMES, feature_by_pair[pair], strict=True)
                    },
                }
            )
    return {"counts": dict(sorted(counts.items())), "examples": dict(sorted(examples.items()))}
