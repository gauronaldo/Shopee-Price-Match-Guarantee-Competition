"""Phase 6 hard-negative mining and mixed-loss invariants."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch.nn import functional as F

from shopee_match.evaluation.protocol import CorpusItem
from shopee_match.models import LearnedMultimodalFusion, MultimodalFusionSpec
from shopee_match.training.hard_negative_data import (
    HardNegativeBatchProvider,
    MiningCandidate,
    cap_variant_share,
    hard_negative_jsonl,
    load_hard_negative_manifest,
    select_hard_negatives,
    select_query_hard_negatives,
)
from shopee_match.training.hard_negative_miner import exact_topk_cosine_block


def _items() -> tuple[CorpusItem, ...]:
    return (
        CorpusItem("a", "a.jpg", "0000", "coffee 100 g"),
        CorpusItem("b", "b.jpg", "1111", "coffee 200 g"),
        CorpusItem("c", "c.jpg", "0000", "different title"),
        CorpusItem("d", "d.jpg", "2222", "coffee 100 g"),
    )


def test_selection_filters_noise_guards_and_deduplicates_symmetric_pairs() -> None:
    items = _items()
    labels = {"a": "one", "b": "two", "c": "three", "d": "four"}
    candidates = [
        [
            MiningCandidate(0, 2, 0.99, 0.90),  # same pHash
            MiningCandidate(0, 3, 0.98, 0.80),  # exact normalized title
            MiningCandidate(0, 1, 0.97, 0.70),  # eligible quantity conflict
        ],
        [MiningCandidate(1, 0, 0.97, 0.70)],
        [],
        [],
    ]
    pairs, stats = select_hard_negatives(
        candidates,
        items,
        labels,
        negatives_per_query=1,
        minimum_pair_probability=0.2,
        maximum_pair_probability=0.98,
        exclude_same_phash=True,
        exclude_exact_normalized_title=True,
        variant_priority_fraction=1.0,
    )
    assert [(pair.left_posting_id, pair.right_posting_id) for pair in pairs] == [("a", "b")]
    assert pairs[0].variant_conflict is True
    assert stats.excluded_same_phash == 1
    assert stats.excluded_exact_title == 1
    assert stats.symmetric_duplicates_removed == 1


def test_exact_block_topk_matches_hand_computed_cosine() -> None:
    embeddings = torch.tensor([[1.0, 0.0], [0.8, 0.2], [0.0, 1.0]])
    indices, scores = exact_topk_cosine_block(embeddings, 0, 2, candidate_k=1)
    assert indices.tolist() == [[1], [0]]
    assert scores[:, 0].tolist() == pytest.approx([0.9701425, 0.9701425])


def test_variant_priority_is_a_hard_share_cap() -> None:
    items = (
        CorpusItem("a", "a.jpg", "0", "drink 100 ml"),
        CorpusItem("b", "b.jpg", "1", "drink 200 ml"),
        CorpusItem("c", "c.jpg", "2", "drink 300 ml"),
        CorpusItem("d", "d.jpg", "3", "drink 400 ml"),
        CorpusItem("e", "e.jpg", "4", "drink 500 ml"),
    )
    labels = {item.posting_id: item.posting_id for item in items}
    pairs, _stats = select_query_hard_negatives(
        0,
        [
            MiningCandidate(0, index, 0.9 - index / 100, 0.8 - index / 100)
            for index in range(1, 5)
        ],
        items,
        labels,
        negatives_per_query=4,
        minimum_pair_probability=0.2,
        maximum_pair_probability=0.98,
        exclude_same_phash=True,
        exclude_exact_normalized_title=True,
        variant_priority_fraction=0.5,
    )
    assert len(pairs) == 2
    assert all(pair.variant_conflict for pair in pairs)


def test_global_variant_cap_is_preserved_after_symmetric_dedup() -> None:
    items = _items()
    labels = {"a": "one", "b": "two", "c": "three", "d": "four"}
    candidates = [
        [MiningCandidate(0, 1, 0.9, 0.8), MiningCandidate(0, 3, 0.8, 0.7)],
        [],
        [],
        [],
    ]
    pairs, _stats = select_hard_negatives(
        candidates,
        items,
        labels,
        negatives_per_query=2,
        minimum_pair_probability=0.2,
        maximum_pair_probability=0.98,
        exclude_same_phash=False,
        exclude_exact_normalized_title=False,
        variant_priority_fraction=0.5,
    )
    capped, removed = cap_variant_share(pairs, 0.5)
    assert len(capped) == 2
    assert sum(pair.variant_conflict for pair in capped) <= len(capped) * 0.5
    assert removed == 0


def test_manifest_and_batch_sampling_are_deterministic(tmp_path: Path) -> None:
    items = _items()
    labels = {"a": "one", "b": "two", "c": "three", "d": "four"}
    pairs, _stats = select_hard_negatives(
        [[MiningCandidate(0, 1, 0.9, 0.8)], [], [], []],
        items,
        labels,
        negatives_per_query=1,
        minimum_pair_probability=0.2,
        maximum_pair_probability=0.98,
        exclude_same_phash=True,
        exclude_exact_normalized_title=True,
        variant_priority_fraction=1.0,
    )
    path = tmp_path / "pairs.jsonl"
    path.write_text(hard_negative_jsonl(pairs), encoding="utf-8")
    loaded = load_hard_negative_manifest(path)
    provider = HardNegativeBatchProvider(
        loaded,
        tuple(item.posting_id for item in items),
        tuple(labels[item.posting_id] for item in items),
        seed=2026,
    )
    first = provider.sample(2, 3, 4)
    second = provider.sample(2, 3, 4)
    assert torch.equal(first[0], second[0])
    assert torch.equal(first[1], second[1])
    left_index = int(first[0][0])
    right_index = int(first[1][0])
    assert labels[items[left_index].posting_id] != labels[items[right_index].posting_id]


def test_mixed_random_and_hard_pair_loss_has_finite_gradients() -> None:
    model = LearnedMultimodalFusion(
        MultimodalFusionSpec(
            image_embedding_dim=4,
            text_embedding_dim=4,
            fusion_hidden_dim=8,
            joint_embedding_dim=4,
            pair_hidden_dim=4,
            dropout=0.0,
        )
    )
    image = torch.randn(6, 4)
    text = torch.randn(6, 4)
    joint = model(image, text)
    random_logits = model.pair_logits(joint[:2], joint[2:4])
    hard_logits = model.pair_logits(joint[:2], joint[4:6])
    random_loss = F.binary_cross_entropy_with_logits(random_logits, torch.tensor([1.0, 0.0]))
    hard_loss = F.binary_cross_entropy_with_logits(hard_logits, torch.zeros(2))
    (0.5 * random_loss + 0.5 * hard_loss).backward()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
