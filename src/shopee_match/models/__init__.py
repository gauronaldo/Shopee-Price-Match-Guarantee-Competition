"""Scratch and explicitly labeled pretrained model adapters (Phases 3-5 and 9)."""

from shopee_match.models.image_encoder import ImageEncoderSpec, ScratchResidualImageEncoder
from shopee_match.models.text_encoder import ScratchTextCNN, TextEncoderSpec

__all__ = [
    "ImageEncoderSpec",
    "ScratchResidualImageEncoder",
    "ScratchTextCNN",
    "TextEncoderSpec",
]
