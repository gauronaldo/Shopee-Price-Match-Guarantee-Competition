from __future__ import annotations

import torch

from shopee_match.evaluation.protocol import CorpusItem
from shopee_match.features.pair_evidence import (
    PairEvidenceRecord,
    fit_pair_evidence_resources,
    pair_evidence_matrix,
)
from shopee_match.models import ResidualPairEvidenceHead


def test_pair_evidence_is_symmetric_and_preserves_identity_tokens() -> None:
    items = (
        CorpusItem("p1", "p1.jpg", "0000000000000000", "Milk 100 ml pack 2"),
        CorpusItem("p2", "p2.jpg", "0000000000000001", "2 Pack MILK 100ml"),
    )
    resources = fit_pair_evidence_resources(
        items,
        items,
        ngram_range=(2, 3),
        max_features=100,
    )
    forward = PairEvidenceRecord("p1", "p2", 0.7, 0.8)
    backward = PairEvidenceRecord("p2", "p1", 0.7, 0.8)
    matrix = pair_evidence_matrix([forward, backward], resources)
    assert matrix[0].tolist() == matrix[1].tolist()
    assert matrix[0, 6] == 0.0


def test_zero_initialized_residual_head_reproduces_frozen_logits() -> None:
    model = ResidualPairEvidenceHead(torch.zeros(3), torch.ones(3))
    baseline = torch.tensor([-1.0, 0.5])
    evidence = torch.tensor([[0.2, 0.3, 0.4], [0.4, 0.3, 0.2]])
    assert torch.equal(model(baseline, evidence), baseline)
    model(baseline, evidence).sum().backward()
    assert model.correction.weight.grad is not None
