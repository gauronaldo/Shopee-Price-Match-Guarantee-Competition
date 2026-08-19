from __future__ import annotations

import pytest
import torch

from shopee_match.losses import SupervisedContrastiveLoss


def test_supervised_contrastive_prefers_separated_product_pairs() -> None:
    loss = SupervisedContrastiveLoss(temperature=0.1)
    labels = torch.tensor([0, 0, 1, 1])
    separated = torch.tensor([[1.0, 0.0], [0.9, 0.1], [-1.0, 0.0], [-0.9, 0.1]])
    mixed = torch.tensor([[1.0, 0.0], [-1.0, 0.0], [0.9, 0.1], [-0.9, 0.1]])

    assert loss(separated, labels) < loss(mixed, labels)


def test_supervised_contrastive_requires_positive_for_every_anchor() -> None:
    with pytest.raises(ValueError, match="every anchor"):
        SupervisedContrastiveLoss()(torch.eye(3), torch.tensor([0, 1, 2]))


def test_supervised_contrastive_has_finite_gradients() -> None:
    embeddings = torch.randn(6, 5, requires_grad=True)
    labels = torch.tensor([0, 0, 1, 1, 2, 2])
    value = SupervisedContrastiveLoss()(embeddings, labels)
    value.backward()

    assert torch.isfinite(value)
    assert embeddings.grad is not None
    assert torch.isfinite(embeddings.grad).all()
