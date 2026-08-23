"""Label-blind classical evidence for exact-product pair decisions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from shopee_match.evaluation.protocol import CorpusItem
from shopee_match.features.image import phash_distance
from shopee_match.features.text import CharTfidfModel, SparseVector, normalize_title
from shopee_match.training.hard_negative_data import identity_tokens

FloatArray = NDArray[np.float32]

PAIR_EVIDENCE_FEATURES = (
    "baseline_probability",
    "joint_cosine_similarity",
    "phash_similarity",
    "tfidf_similarity",
    "title_token_jaccard",
    "identity_token_jaccard",
    "identity_token_conflict",
    "exact_normalized_title",
    "exact_phash",
    "title_length_ratio",
)


@dataclass(frozen=True, slots=True)
class PairEvidenceRecord:
    left_posting_id: str
    right_posting_id: str
    baseline_probability: float
    joint_cosine_similarity: float


@dataclass(frozen=True, slots=True)
class PairEvidenceResources:
    tfidf: CharTfidfModel
    tfidf_by_id: dict[str, SparseVector]
    item_by_id: dict[str, CorpusItem]


def fit_pair_evidence_resources(
    training_items: tuple[CorpusItem, ...],
    transform_items: tuple[CorpusItem, ...],
    *,
    ngram_range: tuple[int, int],
    max_features: int,
) -> PairEvidenceResources:
    """Fit title statistics on train only, then transform the requested listings."""
    model = CharTfidfModel.fit(training_items, ngram_range, max_features)
    return PairEvidenceResources(
        tfidf=model,
        tfidf_by_id={item.posting_id: model.transform_one(item.title) for item in transform_items},
        item_by_id={item.posting_id: item for item in transform_items},
    )


def restore_pair_evidence_resources(
    training_model: CharTfidfModel,
    items: tuple[CorpusItem, ...],
) -> PairEvidenceResources:
    """Transform listings with an already frozen train-only TF-IDF model."""
    return PairEvidenceResources(
        tfidf=training_model,
        tfidf_by_id={item.posting_id: training_model.transform_one(item.title) for item in items},
        item_by_id={item.posting_id: item for item in items},
    )


def _jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 1.0


def _sparse_dot(left: SparseVector, right: SparseVector) -> float:
    if len(left) > len(right):
        left, right = right, left
    return sum(value * right.get(index, 0.0) for index, value in left.items())


def pair_evidence_matrix(
    records: list[PairEvidenceRecord], resources: PairEvidenceResources
) -> FloatArray:
    """Build deterministic symmetric feature rows without using labels."""
    matrix = np.empty((len(records), len(PAIR_EVIDENCE_FEATURES)), dtype=np.float32)
    for row_index, record in enumerate(records):
        left = resources.item_by_id[record.left_posting_id]
        right = resources.item_by_id[record.right_posting_id]
        left_title = normalize_title(left.title)
        right_title = normalize_title(right.title)
        left_tokens, right_tokens = set(left_title.split()), set(right_title.split())
        left_identity, right_identity = (
            set(identity_tokens(left.title)),
            set(identity_tokens(right.title)),
        )
        title_length = min(len(left_title), len(right_title)) / max(
            len(left_title), len(right_title), 1
        )
        matrix[row_index] = (
            record.baseline_probability,
            record.joint_cosine_similarity,
            1.0 - phash_distance(left.image_phash, right.image_phash) / 64.0,
            _sparse_dot(
                resources.tfidf_by_id[left.posting_id],
                resources.tfidf_by_id[right.posting_id],
            ),
            _jaccard(left_tokens, right_tokens),
            _jaccard(left_identity, right_identity),
            float(bool(left_identity or right_identity) and left_identity != right_identity),
            float(left_title == right_title),
            float(left.image_phash == right.image_phash),
            title_length,
        )
    return matrix
