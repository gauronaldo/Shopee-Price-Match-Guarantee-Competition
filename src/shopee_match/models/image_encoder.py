"""Compact residual image encoder trained only from random initialization."""

from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(frozen=True, slots=True)
class ImageEncoderSpec:
    """Serializable architecture contract for the scratch image encoder."""

    input_channels: int
    stem_width: int
    stage_widths: tuple[int, ...]
    blocks_per_stage: tuple[int, ...]
    embedding_dim: int
    projection_hidden_dim: int

    def validate(self) -> None:
        if self.input_channels <= 0 or self.stem_width <= 0:
            raise ValueError("input_channels and stem_width must be positive")
        if not self.stage_widths or len(self.stage_widths) != len(self.blocks_per_stage):
            raise ValueError("stage_widths and blocks_per_stage must have equal non-zero length")
        if any(width <= 0 for width in self.stage_widths):
            raise ValueError("stage widths must be positive")
        if any(blocks <= 0 for blocks in self.blocks_per_stage):
            raise ValueError("every residual stage must contain at least one block")
        if self.embedding_dim <= 0 or self.projection_hidden_dim <= 0:
            raise ValueError("projection dimensions must be positive")


class ResidualBlock(nn.Module):
    """Two-convolution residual block with an explicit projection shortcut."""

    expansion = 1

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.norm1 = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.norm2 = nn.BatchNorm2d(out_channels)
        if stride != 1 or in_channels != out_channels:
            self.shortcut: nn.Module = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, inputs: Tensor) -> Tensor:
        residual = self.shortcut(inputs)
        features = self.activation(self.norm1(self.conv1(inputs)))
        features = self.norm2(self.conv2(features))
        return self.activation(features + residual)


class ScratchResidualImageEncoder(nn.Module):
    """Residual CNN, global pooling, projection head, and unit-length embedding."""

    initialization_policy = "kaiming_normal_conv_linear; batch_norm_unit_scale; random_only"

    def __init__(self, spec: ImageEncoderSpec) -> None:
        super().__init__()
        spec.validate()
        self.spec = spec
        self.stem = nn.Sequential(
            nn.Conv2d(
                spec.input_channels,
                spec.stem_width,
                kernel_size=5,
                stride=2,
                padding=2,
                bias=False,
            ),
            nn.BatchNorm2d(spec.stem_width),
            nn.ReLU(inplace=True),
        )
        stages: list[nn.Module] = []
        in_channels = spec.stem_width
        for stage_index, (width, block_count) in enumerate(
            zip(spec.stage_widths, spec.blocks_per_stage, strict=True)
        ):
            blocks: list[nn.Module] = []
            for block_index in range(block_count):
                stride = 2 if stage_index > 0 and block_index == 0 else 1
                blocks.append(ResidualBlock(in_channels, width, stride))
                in_channels = width
            stages.append(nn.Sequential(*blocks))
        self.stages = nn.Sequential(*stages)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.projection = nn.Sequential(
            nn.Linear(in_channels, spec.projection_hidden_dim, bias=False),
            nn.BatchNorm1d(spec.projection_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(spec.projection_hidden_dim, spec.embedding_dim),
        )
        self.apply(self._initialize_randomly)

    @staticmethod
    def _initialize_randomly(module: nn.Module) -> None:
        if isinstance(module, nn.Conv2d | nn.Linear):
            nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.BatchNorm1d | nn.BatchNorm2d):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward_features(self, inputs: Tensor) -> Tensor:
        features = self.stem(inputs)
        features = self.stages(features)
        return self.pool(features).flatten(1)

    def forward(self, inputs: Tensor) -> Tensor:
        projected = self.projection(self.forward_features(inputs))
        return F.normalize(projected, p=2, dim=1)

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def tensor_shapes(self, image_size: int) -> dict[str, tuple[int, int, int]]:
        """Return channel/height/width contracts for a square input."""
        if image_size <= 0:
            raise ValueError("image_size must be positive")
        spatial = (image_size + 1) // 2
        shapes = {"stem": (self.spec.stem_width, spatial, spatial)}
        for index, width in enumerate(self.spec.stage_widths):
            if index > 0:
                spatial = (spatial + 1) // 2
            shapes[f"stage_{index + 1}"] = (width, spatial, spatial)
        shapes["embedding"] = (self.spec.embedding_dim, 1, 1)
        return shapes
