from __future__ import annotations

import numpy as np

from shopee_match.evaluation.protocol import CorpusItem, ScoredCandidate
from shopee_match.features.pair import (
    FEATURE_NAMES,
    ClassicalPairModel,
    extract_quantities,
    pair_feature_values,
)
from shopee_match.retrieval.pair_matcher import candidate_ceiling


def item(posting_id: str, title: str) -> CorpusItem:
    return CorpusItem(posting_id, f"{posting_id}.jpg", "0", title)


def test_quantity_extraction_normalizes_units() -> None:
    assert extract_quantities("Coffee 0.5 kg") == extract_quantities("Coffee 500 gr")
    assert extract_quantities("Drink 0.5 liter") == extract_quantities("Drink 500 ml")
    assert extract_quantities("Mask pack of 6") == extract_quantities("Mask 6 pcs")


def test_pair_features_expose_numeric_and_quantity_conflicts() -> None:
    values = pair_feature_values(
        item("left", "Phone A52 128GB pack 2"),
        item("right", "Phone A52 256GB pack 6"),
        tfidf_similarity=0.8,
        phash_similarity=0.9,
        orb_similarity=0.2,
    )
    features = dict(zip(FEATURE_NAMES, values, strict=True))

    assert features["digit_conflict"] == 1.0
    assert features["quantity_conflict"] == 1.0
    assert features["model_token_jaccard"] > 0.0


def test_pair_model_is_deterministic_and_serializes_coefficients() -> None:
    negative = np.zeros((3, len(FEATURE_NAMES)), dtype=np.float64)
    positive = np.ones((3, len(FEATURE_NAMES)), dtype=np.float64)
    features = np.vstack([negative, positive])
    labels = np.asarray([0, 0, 0, 1, 1, 1], dtype=np.int64)

    first = ClassicalPairModel.fit(features, labels, 1.0, 2026)
    second = ClassicalPairModel.fit(features, labels, 1.0, 2026)

    np.testing.assert_allclose(first.predict_scores(features), second.predict_scores(features))
    assert set(first.diagnostics()["standardized_coefficients"]) == set(FEATURE_NAMES)


def test_candidate_ceiling_counts_all_union_candidates() -> None:
    candidates = {
        "a": [ScoredCandidate("b", 0.0), ScoredCandidate("c", 0.0)],
        "b": [ScoredCandidate("a", 0.0)],
        "c": [ScoredCandidate("a", 0.0)],
        "d": [ScoredCandidate("c", 0.0)],
    }
    labels = {"a": "first", "b": "first", "c": "second", "d": "second"}

    result = candidate_ceiling(candidates, labels)

    assert result["macro_recall"] == 3 / 4
    assert result["hit_rate"] == 3 / 4
