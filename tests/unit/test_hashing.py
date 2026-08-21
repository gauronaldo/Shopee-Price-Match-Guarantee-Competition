"""Portable frozen-artifact hashing behavior."""

from __future__ import annotations

import hashlib
from pathlib import Path

from shopee_match.hashing import matches_frozen_sha256


def test_text_hash_accepts_line_ending_only_difference(tmp_path: Path) -> None:
    config = tmp_path / "experiment.yaml"
    config.write_bytes(b"seed: 2026\r\nvalue: fixed\r\n")
    expected = hashlib.sha256(b"seed: 2026\nvalue: fixed\n").hexdigest()
    matches, actual = matches_frozen_sha256(config, expected)
    assert matches is True
    assert actual != expected


def test_binary_hash_remains_byte_exact(tmp_path: Path) -> None:
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"binary\r\npayload")
    normalized_hash = hashlib.sha256(b"binary\npayload").hexdigest()
    matches, _actual = matches_frozen_sha256(checkpoint, normalized_hash)
    assert matches is False
