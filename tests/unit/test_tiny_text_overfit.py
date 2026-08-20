from __future__ import annotations

import torch

from shopee_match.losses import SupervisedContrastiveLoss
from shopee_match.models import ScratchTextCNN, TextEncoderSpec
from shopee_match.training.text_data import CharacterVocabulary


def test_text_encoder_overfits_a_hand_checkable_batch() -> None:
    torch.manual_seed(2026)
    titles = (
        "alpha coffee 500g",
        "alpha roasted coffee 500 gram",
        "beta soap 2pcs",
        "beta bath soap pack 2",
    )
    vocabulary = CharacterVocabulary.fit(titles, minimum_frequency=1, maximum_size=64)
    encoded = [vocabulary.encode(title, maximum_length=40) for title in titles]
    token_ids = torch.stack([value[0] for value in encoded])
    lengths = torch.stack([value[1] for value in encoded])
    labels = torch.tensor([0, 0, 1, 1])
    model = ScratchTextCNN(
        len(vocabulary.tokens),
        TextEncoderSpec(12, 12, (3, 5), 16, 8, 0.0),
    )
    loss_function = SupervisedContrastiveLoss(0.1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    initial = float(loss_function(model(token_ids, lengths), labels).detach())
    for _ in range(80):
        optimizer.zero_grad(set_to_none=True)
        loss = loss_function(model(token_ids, lengths), labels)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.inference_mode():
        embeddings = model(token_ids, lengths)
        final = float(loss_function(embeddings, labels))
        similarities = embeddings @ embeddings.T
        similarities.fill_diagonal_(-1)
        nearest = similarities.argmax(dim=1)

    assert final < initial * 0.1
    assert torch.equal(labels[nearest], labels)
