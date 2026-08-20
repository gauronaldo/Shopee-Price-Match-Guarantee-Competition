"""Train-only character vocabulary and product-aware title dataset."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import torch
from torch import Tensor
from torch.utils.data import Dataset

from shopee_match.evaluation.protocol import EvaluationSplit
from shopee_match.features.text import normalize_title

PAD_TOKEN = "<pad>"
UNKNOWN_TOKEN = "<unk>"


@dataclass(frozen=True, slots=True)
class CharacterVocabulary:
    """Character IDs fitted only from normalized training titles."""

    tokens: tuple[str, ...]

    def __post_init__(self) -> None:
        if len(self.tokens) < 3 or self.tokens[:2] != (PAD_TOKEN, UNKNOWN_TOKEN):
            raise ValueError("vocabulary must start with PAD, UNK, and contain a character")
        if len(set(self.tokens)) != len(self.tokens):
            raise ValueError("vocabulary tokens must be unique")

    @classmethod
    def fit(
        cls,
        titles: tuple[str, ...],
        *,
        minimum_frequency: int,
        maximum_size: int,
    ) -> CharacterVocabulary:
        if minimum_frequency <= 0:
            raise ValueError("minimum_frequency must be positive")
        if maximum_size < 3:
            raise ValueError("maximum_size must be at least three")
        counts: Counter[str] = Counter()
        for title in titles:
            counts.update(normalize_title(title))
        selected = sorted(
            (character for character, count in counts.items() if count >= minimum_frequency),
            key=lambda character: (-counts[character], character),
        )[: maximum_size - 2]
        if not selected:
            raise ValueError("training titles do not produce a usable character vocabulary")
        return cls((PAD_TOKEN, UNKNOWN_TOKEN, *selected))

    @property
    def padding_index(self) -> int:
        return 0

    @property
    def unknown_index(self) -> int:
        return 1

    def encode(self, title: str, maximum_length: int) -> tuple[Tensor, Tensor]:
        if maximum_length <= 0:
            raise ValueError("maximum_length must be positive")
        lookup = {token: index for index, token in enumerate(self.tokens)}
        normalized = normalize_title(title)
        values = [lookup.get(character, self.unknown_index) for character in normalized][
            :maximum_length
        ]
        if not values:
            values = [self.unknown_index]
        length = len(values)
        values.extend([self.padding_index] * (maximum_length - length))
        return torch.tensor(values, dtype=torch.long), torch.tensor(length, dtype=torch.long)

    def to_dict(self) -> dict[str, object]:
        return {
            "version": "phase4.character_vocabulary.v1",
            "tokens": list(self.tokens),
            "size": len(self.tokens),
            "padding_index": self.padding_index,
            "unknown_index": self.unknown_index,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> CharacterVocabulary:
        """Restore a checkpoint vocabulary after strict version and token validation."""
        if payload.get("version") != "phase4.character_vocabulary.v1":
            raise ValueError("unsupported character vocabulary version")
        raw_tokens = payload.get("tokens")
        if not isinstance(raw_tokens, list) or not all(
            isinstance(token, str) for token in raw_tokens
        ):
            raise ValueError("vocabulary tokens must be a list of strings")
        return cls(tuple(raw_tokens))


class ProductTextDataset(Dataset[dict[str, Tensor | str]]):
    """Titles selected by a frozen split and encoded with a train-only vocabulary."""

    def __init__(
        self,
        split: EvaluationSplit,
        vocabulary: CharacterVocabulary,
        maximum_length: int,
    ) -> None:
        self.items = split.items
        self.vocabulary = vocabulary
        self.maximum_length = maximum_length
        self.labels = tuple(split.label_by_id[item.posting_id] for item in self.items)
        label_to_index = {label: index for index, label in enumerate(sorted(set(self.labels)))}
        self.label_indices = tuple(label_to_index[label] for label in self.labels)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, Tensor | str]:
        item = self.items[index]
        token_ids, length = self.vocabulary.encode(item.title, self.maximum_length)
        return {
            "token_ids": token_ids,
            "length": length,
            "label": torch.tensor(self.label_indices[index], dtype=torch.long),
            "posting_id": item.posting_id,
        }
