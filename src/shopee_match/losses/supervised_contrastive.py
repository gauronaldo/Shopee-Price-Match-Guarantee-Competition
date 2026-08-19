"""Supervised contrastive objective for product-aware image batches."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class SupervisedContrastiveLoss(nn.Module):
    """Pull same-product embeddings together and push different products apart."""

    def __init__(self, temperature: float = 0.07) -> None:
        super().__init__()
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.temperature = temperature

    def forward(self, embeddings: Tensor, labels: Tensor) -> Tensor:
        if embeddings.ndim != 2:
            raise ValueError("embeddings must have shape [batch, dimension]")
        if labels.ndim != 1 or labels.shape[0] != embeddings.shape[0]:
            raise ValueError("labels must have shape [batch]")
        if embeddings.shape[0] < 2:
            raise ValueError("supervised contrastive loss requires at least two samples")

        normalized = F.normalize(embeddings, p=2, dim=1)
        logits = normalized @ normalized.T / self.temperature
        self_mask = torch.eye(logits.shape[0], dtype=torch.bool, device=logits.device)
        positive_mask = labels[:, None].eq(labels[None, :]) & ~self_mask
        positive_counts = positive_mask.sum(dim=1)
        if torch.any(positive_counts == 0):
            raise ValueError("every anchor must have another sample with the same label")

        logits = logits - logits.max(dim=1, keepdim=True).values.detach()
        denominator_logits = logits.masked_fill(self_mask, float("-inf"))
        log_denominator = torch.logsumexp(denominator_logits, dim=1, keepdim=True)
        log_probabilities = logits - log_denominator
        positive_log_prob = log_probabilities.masked_fill(~positive_mask, 0.0).sum(dim=1)
        return -(positive_log_prob / positive_counts).mean()
