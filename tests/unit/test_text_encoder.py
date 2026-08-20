from __future__ import annotations

import torch

from shopee_match.losses import SupervisedContrastiveLoss
from shopee_match.models import ScratchTextCNN, TextEncoderSpec


def _spec() -> TextEncoderSpec:
    return TextEncoderSpec(
        character_embedding_dim=8,
        convolution_channels=6,
        kernel_sizes=(3, 5),
        projection_hidden_dim=10,
        embedding_dim=7,
        dropout=0.0,
    )


def test_text_cnn_shape_norm_gradient_and_serialization() -> None:
    torch.manual_seed(7)
    model = ScratchTextCNN(20, _spec())
    token_ids = torch.randint(2, 20, (4, 12))
    lengths = torch.tensor([12, 10, 8, 6])
    labels = torch.tensor([0, 0, 1, 1])

    embeddings = model(token_ids, lengths)
    SupervisedContrastiveLoss(0.1)(embeddings, labels).backward()

    assert embeddings.shape == (4, 7)
    assert torch.isfinite(embeddings).all()
    assert torch.allclose(torch.linalg.vector_norm(embeddings, dim=1), torch.ones(4), atol=1e-5)
    assert all(parameter.grad is not None for parameter in model.parameters())

    clone = ScratchTextCNN(20, _spec())
    clone.load_state_dict(model.state_dict())
    model.eval()
    clone.eval()
    with torch.inference_mode():
        assert torch.equal(model(token_ids, lengths), clone(token_ids, lengths))


def test_text_encoder_initialization_is_seeded_and_random() -> None:
    torch.manual_seed(10)
    first = ScratchTextCNN(20, _spec())
    torch.manual_seed(10)
    second = ScratchTextCNN(20, _spec())
    torch.manual_seed(11)
    third = ScratchTextCNN(20, _spec())

    assert torch.equal(first.character_embedding.weight, second.character_embedding.weight)
    assert not torch.equal(first.character_embedding.weight, third.character_embedding.weight)
    assert torch.count_nonzero(first.character_embedding.weight[0]) == 0
