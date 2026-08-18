from __future__ import annotations

from shopee_match.evaluation.protocol import CorpusItem, ScoredCandidate
from shopee_match.features.image import phash_distance, rank_phash
from shopee_match.features.text import CharTfidfModel, normalize_title
from shopee_match.retrieval.fusion import candidate_union, fuse_rankings


def item(posting_id: str, phash: str, title: str) -> CorpusItem:
    return CorpusItem(posting_id, f"{posting_id}.jpg", phash, title)


def test_title_normalization_preserves_variant_digits_and_units() -> None:
    assert normalize_title("  CÀ-PHÊ 500g / 0.5 KG ") == "cà phê 500g 0 5 kg"


def test_tfidf_vocabulary_is_fit_only_from_supplied_training_items() -> None:
    training = (item("train", "0", "coffee 500g"),)
    model = CharTfidfModel.fit(training, (3, 3), 100)

    assert model.transform_one("coffee 500g")
    assert not model.transform_one("xyz")


def test_phash_ranking_is_deterministic_on_ties() -> None:
    items = (
        item("q", "0000000000000000", "q"),
        item("b", "0000000000000001", "b"),
        item("a", "0000000000000002", "a"),
    )

    ranking = rank_phash(items, 2)

    assert phash_distance(items[0].image_phash, items[1].image_phash) == 1
    assert [candidate.posting_id for candidate in ranking["q"]] == ["a", "b"]


def test_candidate_union_and_fusion_are_label_blind_and_deterministic() -> None:
    visual = {
        "q": [ScoredCandidate("a", 0.8), ScoredCandidate("b", 0.2)],
        "a": [ScoredCandidate("q", 0.8), ScoredCandidate("b", 0.1)],
        "b": [ScoredCandidate("q", 0.2), ScoredCandidate("a", 0.1)],
    }
    text = {
        "q": [ScoredCandidate("b", 0.9), ScoredCandidate("a", 0.1)],
        "a": [ScoredCandidate("b", 0.7), ScoredCandidate("q", 0.1)],
        "b": [ScoredCandidate("q", 0.9), ScoredCandidate("a", 0.7)],
    }

    union = candidate_union((visual, text), 1)
    fused = fuse_rankings(visual, text, 0.5, 2)

    assert [candidate.posting_id for candidate in union["q"]] == ["a", "b"]
    assert [candidate.posting_id for candidate in fused["q"]] == ["b", "a"]
