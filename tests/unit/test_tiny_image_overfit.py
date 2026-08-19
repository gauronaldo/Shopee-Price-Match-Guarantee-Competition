from __future__ import annotations

import torch

from shopee_match.losses import SupervisedContrastiveLoss
from shopee_match.models import ImageEncoderSpec, ScratchResidualImageEncoder


def test_scratch_encoder_overfits_one_hand_checkable_batch() -> None:
    torch.manual_seed(2026)
    first_product = torch.randn(1, 3, 24, 24)
    second_product = torch.randn(1, 3, 24, 24)
    inputs = torch.cat(
        [
            first_product,
            first_product + 0.01 * torch.randn_like(first_product),
            second_product,
            second_product + 0.01 * torch.randn_like(second_product),
        ]
    )
    labels = torch.tensor([0, 0, 1, 1])
    model = ScratchResidualImageEncoder(
        ImageEncoderSpec(3, 4, (4, 8), (1, 1), embedding_dim=8, projection_hidden_dim=8)
    )
    objective = SupervisedContrastiveLoss(temperature=0.1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    initial = float(objective(model(inputs), labels).detach())
    for _ in range(25):
        optimizer.zero_grad(set_to_none=True)
        loss = objective(model(inputs), labels)
        loss.backward()
        optimizer.step()
    final = float(objective(model(inputs), labels).detach())

    assert final < initial * 0.5
