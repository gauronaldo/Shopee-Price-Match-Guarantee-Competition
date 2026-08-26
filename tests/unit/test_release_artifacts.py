from __future__ import annotations

from pathlib import Path

import yaml

from shopee_match.hashing import sha256_file
from shopee_match.serving.cli import _parser
from shopee_match.serving.release_artifacts import (
    build_release_archive,
    install_model_release,
    load_model_release,
    model_release_status,
)


def _manifest(path: Path, file_sha: str, asset_sha: str, size: int) -> None:
    payload = {
        "manifest_version": "demo.model_release.v1",
        "release": {
            "repository": "owner/project",
            "tag": "v1.0.0",
            "asset": "models.zip",
            "asset_sha256": asset_sha,
        },
        "split_manifest_sha256": "1" * 64,
        "files": [
            {
                "path": "artifacts/model/best.pt",
                "sha256": file_sha,
                "size_bytes": size,
            }
        ],
    }
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def test_release_archive_round_trip_is_verified(tmp_path: Path) -> None:
    source = tmp_path / "artifacts" / "model" / "best.pt"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"frozen-model-weights")
    manifest = tmp_path / "release.yaml"
    archive = tmp_path / "models.zip"
    _manifest(manifest, sha256_file(source), "0" * 64, source.stat().st_size)

    packaged = build_release_archive(manifest, archive, root=tmp_path)
    _manifest(manifest, sha256_file(source), str(packaged["sha256"]), source.stat().st_size)
    source.unlink()

    assert model_release_status(manifest, tmp_path)["status"] == "incomplete"
    installed = install_model_release(manifest, root=tmp_path, url=archive.as_uri())
    assert installed["status"] == "ready"
    assert installed["downloaded"] is True
    assert source.read_bytes() == b"frozen-model-weights"


def test_project_release_manifest_is_strict_and_portable() -> None:
    release = load_model_release(Path("configs/serving/model_release.yaml"))
    assert release.repository == "gauronaldo/Shopee-Price-Match-Guarantee-Competition"
    assert release.download_url.endswith("/v1.0.0/shopee_demo_models_v1.0.0.zip")
    assert all(not item.path.is_absolute() for item in release.files)
    assert all("mined_pairs" not in item.path.as_posix() for item in release.files)


def test_demo_cli_exposes_release_and_bootstrap_commands() -> None:
    assert _parser().parse_args(["models-status"]).command == "models-status"
    assert _parser().parse_args(["download-models"]).command == "download-models"
    bootstrap = _parser().parse_args(["bootstrap", "--device", "cpu"])
    assert bootstrap.command == "bootstrap"
    assert bootstrap.device == "cpu"
