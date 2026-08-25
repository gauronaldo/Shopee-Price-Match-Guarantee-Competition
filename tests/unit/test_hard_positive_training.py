from __future__ import annotations

import numpy as np
import torch

from shopee_match.models.multimodal_fusion import (
    LearnedMultimodalFusion,
    MultimodalFusionSpec,
)
from shopee_match.training.hard_positive_config import HardPositiveTrainingConfig
from shopee_match.training.hard_positive_trainer import (
    MixedPairBatchProvider,
    deterministic_group_holdout,
    select_hard_positive_pairs,
)


def _training_config() -> HardPositiveTrainingConfig:
    return HardPositiveTrainingConfig(
        device="cpu",
        epochs=1,
        batches_per_epoch=1,
        batch_size=8,
        learning_rate=0.001,
        weight_decay=0.0,
        gradient_clip_norm=1.0,
        early_stopping_patience=1,
        hard_positive_fraction=0.25,
        hard_negative_fraction=0.25,
        random_positive_fraction=0.25,
        random_negative_fraction=0.25,
        trainable_components="pair_head",
        positive_class_weight=1.0,
    )


def test_group_holdout_is_deterministic_and_group_disjoint() -> None:
    labels = tuple(label for group in range(12) for label in (f"g{group}", f"g{group}"))

    first = deterministic_group_holdout(labels, holdout_fraction=0.25, seed=17)
    second = deterministic_group_holdout(labels, holdout_fraction=0.25, seed=17)

    assert first == second
    optimization_labels = {labels[index] for index in first.optimization_indices}
    holdout_labels = {labels[index] for index in first.holdout_indices}
    assert optimization_labels.isdisjoint(holdout_labels)
    assert len(holdout_labels) == 3


def test_hard_positive_selection_uses_lowest_probability_then_pair_order() -> None:
    pairs = [(0, 1), (2, 3), (4, 5), (6, 7)]
    probabilities = np.asarray([0.8, 0.1, 0.1, 0.4], dtype=np.float32)

    selected = select_hard_positive_pairs(pairs, probabilities, limit=3)

    assert selected == [(2, 3), (4, 5), (6, 7)]


def test_mixed_pair_batch_is_deterministic_and_balanced() -> None:
    labels = ("a", "a", "b", "b", "c", "c", "d", "d")
    provider = MixedPairBatchProvider(
        hard_positives=[(0, 1)],
        hard_negatives=[(0, 2)],
        random_positives=[(2, 3), (4, 5), (6, 7)],
        allowed_indices=tuple(range(8)),
        labels=labels,
        seed=23,
    )
    config = _training_config()

    first = provider.sample(epoch=0, batch_index=0, batch_size=8, config=config)
    second = provider.sample(epoch=0, batch_index=0, batch_size=8, config=config)

    assert all(torch.equal(left, right) for left, right in zip(first, second, strict=True))
    left, right, targets = first
    assert int(targets.sum().item()) == 4
    assert all(
        labels[left[index]] == labels[right[index]]
        for index in torch.nonzero(targets == 1, as_tuple=False).flatten().tolist()
    )
    assert all(
        labels[left[index]] != labels[right[index]]
        for index in torch.nonzero(targets == 0, as_tuple=False).flatten().tolist()
    )


def test_recall_refinement_can_update_fusion_and_pair_head() -> None:
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
    image = torch.randn(4, 4)
    text = torch.randn(4, 4)
    joint = model(image, text)
    logits = model.pair_logits(joint[:2], joint[2:])
    loss = torch.nn.functional.binary_cross_entropy_with_logits(
        logits,
        torch.ones(2),
        pos_weight=torch.tensor(1.25),
    )
    loss.backward()

    assert any(parameter.grad is not None for parameter in model.fusion.parameters())
    assert any(parameter.grad is not None for parameter in model.pair_head.parameters())
