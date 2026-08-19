from __future__ import annotations

import torch

from shopee_match.losses import SupervisedContrastiveLoss
from shopee_match.models import ImageEncoderSpec, ScratchResidualImageEncoder


def _spec() -> ImageEncoderSpec:
    return ImageEncoderSpec(
        input_channels=3,
        stem_width=8,
        stage_widths=(8, 16),
        blocks_per_stage=(1, 1),
        embedding_dim=12,
        projection_hidden_dim=16,
    )


def test_scratch_image_encoder_shape_norm_gradient_and_serialization() -> None:
    torch.manual_seed(7)
    model = ScratchResidualImageEncoder(_spec())
    inputs = torch.randn(4, 3, 32, 32)
    labels = torch.tensor([0, 0, 1, 1])

    embeddings = model(inputs)
    loss = SupervisedContrastiveLoss(temperature=0.1)(embeddings, labels)
    loss.backward()

    assert embeddings.shape == (4, 12)
    assert model.tensor_shapes(32) == {
        "stem": (8, 16, 16),
        "stage_1": (8, 16, 16),
        "stage_2": (16, 8, 8),
        "embedding": (12, 1, 1),
    }
    assert torch.isfinite(embeddings).all()
    assert torch.allclose(torch.linalg.vector_norm(embeddings, dim=1), torch.ones(4), atol=1e-5)
    assert all(parameter.grad is not None for parameter in model.parameters())
    assert all(
        torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
        if parameter.grad is not None
    )

    clone = ScratchResidualImageEncoder(_spec())
    clone.load_state_dict(model.state_dict())
    model.eval()
    clone.eval()
    with torch.inference_mode():
        assert torch.equal(model(inputs), clone(inputs))


def test_encoder_initialization_is_seeded_and_random() -> None:
    torch.manual_seed(10)
    first = ScratchResidualImageEncoder(_spec())
    torch.manual_seed(10)
    second = ScratchResidualImageEncoder(_spec())
    torch.manual_seed(11)
    third = ScratchResidualImageEncoder(_spec())

    first_weight = next(first.parameters())
    assert torch.equal(first_weight, next(second.parameters()))
    assert not torch.equal(first_weight, next(third.parameters()))
    assert "random_only" in first.initialization_policy
