from __future__ import annotations

import torch

from shopee_match.features.text import normalize_title
from shopee_match.training.text_data import CharacterVocabulary


def test_normalization_preserves_identity_critical_digits_and_units() -> None:
    normalized = normalize_title("Brand-X 12.5 ML / Pack 03")

    assert "12" in normalized
    assert "5" in normalized
    assert "ml" in normalized
    assert "03" in normalized


def test_vocabulary_is_deterministic_and_unknown_characters_do_not_leak() -> None:
    titles = ("alpha 500g", "alpha 500g", "beta 2pcs")
    first = CharacterVocabulary.fit(titles, minimum_frequency=1, maximum_size=64)
    second = CharacterVocabulary.fit(titles, minimum_frequency=1, maximum_size=64)

    token_ids, length = first.encode("alpha Ω 500g", maximum_length=20)

    assert first == second
    assert first.unknown_index in token_ids.tolist()
    assert int(length) == len(normalize_title("alpha Ω 500g"))
    assert torch.all(token_ids[int(length) :] == first.padding_index)
