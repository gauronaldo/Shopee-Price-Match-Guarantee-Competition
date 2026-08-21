"""Repository-owned learned fusion and symmetric pair scoring for Phase 5."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(frozen=True, slots=True)
class MultimodalFusionSpec:
    """Architecture contract for fusing frozen scratch modality embeddings."""

    image_embedding_dim: int
    text_embedding_dim: int
    fusion_hidden_dim: int
    joint_embedding_dim: int
    pair_hidden_dim: int
    dropout: float
    fusion_mode: str = "projected"
    base_image_weight: float = 0.5
    residual_scale: float = 1.0

    def validate(self) -> None:
        dimensions = (
            self.image_embedding_dim,
            self.text_embedding_dim,
            self.fusion_hidden_dim,
            self.joint_embedding_dim,
            self.pair_hidden_dim,
        )
        if min(dimensions) <= 0:
            raise ValueError("multimodal dimensions must be positive")
        if self.image_embedding_dim != self.text_embedding_dim:
            raise ValueError("image and text embeddings must have equal dimensions")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        if self.fusion_mode not in {"projected", "residual_score_preserving"}:
            raise ValueError("unsupported fusion_mode")
        if not 0 <= self.base_image_weight <= 1:
            raise ValueError("base_image_weight must be inside [0, 1]")
        if self.residual_scale <= 0:
            raise ValueError("residual_scale must be positive")
        if self.fusion_mode == "residual_score_preserving" and self.joint_embedding_dim != (
            self.image_embedding_dim + self.text_embedding_dim
        ):
            raise ValueError("score-preserving residual fusion requires concatenated joint size")


class LearnedMultimodalFusion(nn.Module):
    """Fuse image/text embeddings and optionally score symmetric listing pairs."""

    initialization_policy = "kaiming_normal_linear; layer_norm_unit_scale; random_only"

    def __init__(self, spec: MultimodalFusionSpec) -> None:
        super().__init__()
        spec.validate()
        self.spec = spec
        feature_dim = spec.image_embedding_dim * 4
        self.fusion = nn.Sequential(
            nn.Linear(feature_dim, spec.fusion_hidden_dim),
            nn.LayerNorm(spec.fusion_hidden_dim),
            nn.GELU(),
            nn.Dropout(spec.dropout),
            nn.Linear(spec.fusion_hidden_dim, spec.joint_embedding_dim),
        )
        self.pair_head = nn.Sequential(
            nn.Linear(spec.joint_embedding_dim * 2, spec.pair_hidden_dim),
            nn.GELU(),
            nn.Dropout(spec.dropout),
            nn.Linear(spec.pair_hidden_dim, 1),
        )
        self.reset_parameters()
        if spec.fusion_mode == "residual_score_preserving":
            output = self.fusion[-1]
            if not isinstance(output, nn.Linear):  # pragma: no cover - architecture invariant
                raise AssertionError("fusion output must be linear")
            nn.init.zeros_(output.weight)
            nn.init.zeros_(output.bias)

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity="linear")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, image_embeddings: Tensor, text_embeddings: Tensor) -> Tensor:
        if image_embeddings.ndim != 2 or text_embeddings.ndim != 2:
            raise ValueError("modality embeddings must have shape [batch, dimension]")
        if image_embeddings.shape[0] != text_embeddings.shape[0]:
            raise ValueError("image and text batch sizes must match")
        expected = (self.spec.image_embedding_dim, self.spec.text_embedding_dim)
        if (image_embeddings.shape[1], text_embeddings.shape[1]) != expected:
            raise ValueError(f"modality dimensions must be {expected}")
        image = F.normalize(image_embeddings, p=2, dim=1)
        text = F.normalize(text_embeddings, p=2, dim=1)
        features = torch.cat((image, text, image * text, torch.abs(image - text)), dim=1)
        correction = self.fusion(features)
        if self.spec.fusion_mode == "projected":
            return F.normalize(correction, p=2, dim=1)
        image_scale = self.spec.base_image_weight**0.5
        text_scale = (1 - self.spec.base_image_weight) ** 0.5
        base = torch.cat((image_scale * image, text_scale * text), dim=1)
        return F.normalize(base + self.spec.residual_scale * correction, p=2, dim=1)

    def pair_logits(self, left: Tensor, right: Tensor) -> Tensor:
        """Return order-invariant pair logits from normalized joint embeddings."""
        if left.ndim != 2 or right.ndim != 2 or left.shape != right.shape:
            raise ValueError("pair embeddings must have equal [pairs, dimension] shapes")
        if left.shape[1] != self.spec.joint_embedding_dim:
            raise ValueError("pair embedding dimension differs from the fusion contract")
        features = torch.cat((left * right, torch.abs(left - right)), dim=1)
        return cast(Tensor, self.pair_head(features).squeeze(1))

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


def balanced_pair_indices(
    labels: Tensor,
    *,
    maximum_negative_ratio: int,
    generator: torch.Generator | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    """Select all in-batch positives and a bounded deterministic negative subset."""
    if labels.ndim != 1:
        raise ValueError("labels must have shape [batch]")
    if maximum_negative_ratio <= 0:
        raise ValueError("maximum_negative_ratio must be positive")
    pairs = torch.triu_indices(labels.numel(), labels.numel(), offset=1, device=labels.device)
    left, right = pairs[0], pairs[1]
    targets = labels[left].eq(labels[right])
    positive_indices = torch.nonzero(targets, as_tuple=False).flatten()
    negative_indices = torch.nonzero(~targets, as_tuple=False).flatten()
    if positive_indices.numel() == 0:
        raise ValueError("batch contains no positive pair")
    maximum_negatives = positive_indices.numel() * maximum_negative_ratio
    if negative_indices.numel() > maximum_negatives:
        order = torch.randperm(negative_indices.numel(), generator=generator, device="cpu")
        chosen = order[:maximum_negatives].to(negative_indices.device)
        negative_indices = negative_indices[chosen]
    selected = torch.cat((positive_indices, negative_indices))
    selected, _ = torch.sort(selected)
    return left[selected], right[selected], targets[selected].to(dtype=torch.float32)
