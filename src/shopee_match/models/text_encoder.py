"""Character-level title encoder trained from random initialization."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(frozen=True, slots=True)
class TextEncoderSpec:
    """Architecture contract independent of the train-only vocabulary size."""

    character_embedding_dim: int
    convolution_channels: int
    kernel_sizes: tuple[int, ...]
    projection_hidden_dim: int
    embedding_dim: int
    dropout: float

    def validate(self) -> None:
        if (
            min(
                self.character_embedding_dim,
                self.convolution_channels,
                self.projection_hidden_dim,
                self.embedding_dim,
            )
            <= 0
        ):
            raise ValueError("text encoder dimensions must be positive")
        if not self.kernel_sizes or any(size <= 0 or size % 2 == 0 for size in self.kernel_sizes):
            raise ValueError("kernel_sizes must be non-empty positive odd integers")
        if len(set(self.kernel_sizes)) != len(self.kernel_sizes):
            raise ValueError("kernel_sizes must be unique")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")


class ScratchTextCNN(nn.Module):
    """Multi-kernel character CNN producing normalized product-title embeddings."""

    initialization_policy = (
        "random_normal_embedding; kaiming_normal_conv_linear; layer_norm_unit_scale; random_only"
    )

    def __init__(
        self, vocabulary_size: int, spec: TextEncoderSpec, *, padding_index: int = 0
    ) -> None:
        super().__init__()
        spec.validate()
        if vocabulary_size <= 2:
            raise ValueError("vocabulary_size must include PAD, UNK, and at least one character")
        if not 0 <= padding_index < vocabulary_size:
            raise ValueError("padding_index must be inside the vocabulary")
        self.spec = spec
        self.padding_index = padding_index
        self.character_embedding = nn.Embedding(
            vocabulary_size,
            spec.character_embedding_dim,
            padding_idx=padding_index,
        )
        self.convolutions = nn.ModuleList(
            nn.Conv1d(
                spec.character_embedding_dim,
                spec.convolution_channels,
                kernel_size=size,
                padding=size // 2,
            )
            for size in spec.kernel_sizes
        )
        pooled_dimension = spec.convolution_channels * len(spec.kernel_sizes)
        self.feature_norm = nn.LayerNorm(pooled_dimension)
        self.dropout = nn.Dropout(spec.dropout)
        self.projection = nn.Sequential(
            nn.Linear(pooled_dimension, spec.projection_hidden_dim),
            nn.GELU(),
            nn.Linear(spec.projection_hidden_dim, spec.embedding_dim),
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.character_embedding.weight, mean=0.0, std=0.02)
        with torch.no_grad():
            self.character_embedding.weight[self.padding_index].zero_()
        for convolution in self.convolutions:
            nn.init.kaiming_normal_(convolution.weight, nonlinearity="relu")
            if convolution.bias is not None:
                nn.init.zeros_(convolution.bias)
        nn.init.ones_(self.feature_norm.weight)
        nn.init.zeros_(self.feature_norm.bias)
        for module in self.projection:
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity="linear")
                nn.init.zeros_(module.bias)

    def forward(self, token_ids: Tensor, lengths: Tensor) -> Tensor:
        if token_ids.ndim != 2:
            raise ValueError("token_ids must have shape [batch, sequence]")
        if lengths.ndim != 1 or lengths.shape[0] != token_ids.shape[0]:
            raise ValueError("lengths must have shape [batch]")
        if token_ids.shape[1] < max(self.spec.kernel_sizes):
            raise ValueError("sequence length must cover the largest convolution kernel")
        if torch.any(lengths <= 0) or torch.any(lengths > token_ids.shape[1]):
            raise ValueError("lengths must be within the padded sequence")

        sequence = self.character_embedding(token_ids).transpose(1, 2)
        valid_positions = torch.arange(token_ids.shape[1], device=token_ids.device)[None, :]
        valid_positions = valid_positions < lengths[:, None]
        pooled: list[Tensor] = []
        for convolution in self.convolutions:
            features = F.gelu(convolution(sequence))
            features = features.masked_fill(~valid_positions[:, None, :], float("-inf"))
            pooled.append(features.amax(dim=2))
        combined = self.dropout(self.feature_norm(torch.cat(pooled, dim=1)))
        return F.normalize(self.projection(combined), p=2, dim=1)

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())
