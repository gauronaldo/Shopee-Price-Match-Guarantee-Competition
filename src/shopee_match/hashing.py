"""Portable artifact hashing helpers."""

from __future__ import annotations

import hashlib
from pathlib import Path

TEXT_SUFFIXES = frozenset({".json", ".jsonl", ".md", ".toml", ".txt", ".yaml", ".yml"})


def sha256_file(path: Path) -> str:
    """Hash the exact bytes stored on disk."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_text_sha256(path: Path) -> str:
    """Hash text with LF line endings so Git checkout policy cannot change identity."""
    normalized = path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(normalized).hexdigest()


def matches_frozen_sha256(path: Path, expected: str) -> tuple[bool, str]:
    """Accept exact bytes, or line-ending-only differences for known text artifacts."""
    actual = sha256_file(path)
    if actual == expected:
        return True, actual
    if path.suffix.lower() in TEXT_SUFFIXES and canonical_text_sha256(path) == expected:
        return True, actual
    return False, actual
