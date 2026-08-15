"""Typed records used by Phase 1 ingestion, audit, and splitting."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

Severity = Literal["warning", "critical"]
SplitName = Literal["train", "validation", "test"]


@dataclass(frozen=True, slots=True)
class Listing:
    row_number: int
    posting_id: str
    image: str
    image_phash: str
    title: str
    label_group: str


@dataclass(frozen=True, slots=True)
class ImageRecord:
    image: str
    path: Path
    sha256: str
    width: int
    height: int
    file_bytes: int


@dataclass(frozen=True, slots=True)
class Finding:
    severity: Severity
    code: str
    message: str
    count: int


@dataclass(frozen=True, slots=True)
class Component:
    component_id: str
    label_groups: tuple[str, ...]
    row_count: int
    size_band: str
