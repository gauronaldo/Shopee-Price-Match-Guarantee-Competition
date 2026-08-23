"""Residual pair-evidence head over a frozen learned pair score."""

from __future__ import annotations

from typing import cast

import torch
from torch import Tensor, nn


class ResidualPairEvidenceHead(nn.Module):
    """Learn a small correction while exactly reproducing the frozen score at initialization."""

    feature_mean: Tensor
    feature_scale: Tensor

    def __init__(self, feature_mean: Tensor, feature_scale: Tensor) -> None:
        super().__init__()
        if (
            feature_mean.ndim != 1
            or feature_scale.shape != feature_mean.shape
            or feature_mean.numel() == 0
            or not torch.isfinite(feature_mean).all()
            or not torch.isfinite(feature_scale).all()
            or not torch.all(feature_scale > 0)
        ):
            raise ValueError("feature normalization tensors must be finite positive vectors")
        self.register_buffer("feature_mean", feature_mean.detach().to(dtype=torch.float32))
        self.register_buffer("feature_scale", feature_scale.detach().to(dtype=torch.float32))
        self.correction = nn.Linear(feature_mean.numel(), 1)
        nn.init.zeros_(self.correction.weight)
        nn.init.zeros_(self.correction.bias)

    def forward(self, baseline_logits: Tensor, evidence: Tensor) -> Tensor:
        if baseline_logits.ndim != 1 or evidence.ndim != 2:
            raise ValueError(
                "baseline logits and evidence must have shapes [pairs] and [pairs, features]"
            )
        if evidence.shape != (baseline_logits.shape[0], self.feature_mean.numel()):
            raise ValueError("pair evidence shape differs from the normalization contract")
        normalized = (evidence - self.feature_mean) / self.feature_scale
        return baseline_logits + cast(Tensor, self.correction(normalized).squeeze(1))

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())
