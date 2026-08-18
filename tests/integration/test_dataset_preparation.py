from __future__ import annotations

from pathlib import Path

import pytest

from shopee_match.data.pipeline import prepare_dataset
from shopee_match.errors import DataValidationError, OutputConflictError

from ..dataset_helpers import make_dataset_workspace


def test_dataset_preparation_is_deterministic_and_idempotent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_root = Path.cwd()
    make_dataset_workspace(tmp_path, source_root)
    monkeypatch.chdir(tmp_path)

    first = prepare_dataset(Path("phase1.yaml"))
    manifest_before = Path(first["manifest"]).read_bytes()
    second = prepare_dataset(Path("phase1.yaml"))

    assert first["manifest_sha256"] == second["manifest_sha256"]
    assert Path(second["manifest"]).read_bytes() == manifest_before
    assert first["split"]["integrity"] == {
        "exact_phashes_cross_split": 0,
        "image_references_cross_split": 0,
        "label_groups_cross_split": 0,
        "sha256_cross_split": 0,
    }
    assert len(manifest_before.splitlines()) == 6
    assert Path("outputs/audit.md").exists()
    assert Path("outputs/inspection/gallery.html").exists()


def test_dataset_preparation_blocks_corrupt_images(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_root = Path.cwd()
    make_dataset_workspace(tmp_path, source_root)
    (tmp_path / "raw" / "images" / "red_front.ppm").write_bytes(b"not-an-image")
    monkeypatch.chdir(tmp_path)

    with pytest.raises(DataValidationError, match="image_decode_failure"):
        prepare_dataset(Path("phase1.yaml"))


def test_dataset_preparation_refuses_changed_versioned_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_root = Path.cwd()
    make_dataset_workspace(tmp_path, source_root)
    monkeypatch.chdir(tmp_path)
    prepare_dataset(Path("phase1.yaml"))
    Path("outputs/split.jsonl").write_text("changed\n", encoding="utf-8")

    with pytest.raises(OutputConflictError):
        prepare_dataset(Path("phase1.yaml"))
