"""Download, verify, and package the inference artifacts published with a release."""

from __future__ import annotations

import json
import shutil
import tempfile
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, cast

import yaml

from shopee_match.errors import ConfigurationError, DataValidationError, OutputConflictError
from shopee_match.hashing import sha256_file


@dataclass(frozen=True)
class ReleaseFile:
    """One file restored at its canonical project-relative path."""

    path: Path
    sha256: str
    size_bytes: int


@dataclass(frozen=True)
class ModelRelease:
    """Immutable public release contract used by the demo installer."""

    repository: str
    tag: str
    asset: str
    asset_sha256: str
    split_manifest_sha256: str
    files: tuple[ReleaseFile, ...]

    @property
    def download_url(self) -> str:
        return f"https://github.com/{self.repository}/releases/download/{self.tag}/{self.asset}"


def _mapping(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ConfigurationError(f"{name} must be a mapping")
    return cast(dict[str, Any], value)


def _sha256(value: object, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ConfigurationError(f"{name} must be a lowercase SHA-256 digest")
    try:
        int(value, 16)
    except ValueError as exc:
        raise ConfigurationError(f"{name} must be a lowercase SHA-256 digest") from exc
    if value != value.lower():
        raise ConfigurationError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _relative_path(value: object, name: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ConfigurationError(f"{name} must be a non-empty relative path")
    pure = PurePosixPath(value)
    if pure.is_absolute() or ".." in pure.parts or "." in pure.parts:
        raise ConfigurationError(f"{name} must stay within the repository")
    return Path(*pure.parts)


def load_model_release(path: Path) -> ModelRelease:
    """Load a strict release manifest without touching remote state."""
    try:
        root = _mapping(yaml.safe_load(path.read_text(encoding="utf-8")), "release manifest")
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigurationError(f"Cannot read model release manifest: {path}") from exc
    expected_root = {"manifest_version", "release", "split_manifest_sha256", "files"}
    if set(root) != expected_root or root["manifest_version"] != "demo.model_release.v1":
        raise ConfigurationError("Unsupported model release manifest")
    release = _mapping(root["release"], "release")
    if set(release) != {"repository", "tag", "asset", "asset_sha256"}:
        raise ConfigurationError("release contains unsupported or missing fields")
    repository = release["repository"]
    tag = release["tag"]
    asset = release["asset"]
    if not all(isinstance(value, str) and value.strip() for value in (repository, tag, asset)):
        raise ConfigurationError("release repository, tag, and asset must be non-empty strings")
    if repository.startswith(("/", ".")) or repository.count("/") != 1:
        raise ConfigurationError("release.repository must use the owner/repository form")
    if "/" in asset or "\\" in asset:
        raise ConfigurationError("release.asset must be a filename")
    raw_files = root["files"]
    if not isinstance(raw_files, list) or not raw_files:
        raise ConfigurationError("files must be a non-empty list")
    files: list[ReleaseFile] = []
    seen: set[Path] = set()
    for index, raw_file in enumerate(raw_files):
        item = _mapping(raw_file, f"files[{index}]")
        if set(item) != {"path", "sha256", "size_bytes"}:
            raise ConfigurationError(f"files[{index}] contains unsupported or missing fields")
        file_path = _relative_path(item["path"], f"files[{index}].path")
        size = item["size_bytes"]
        if file_path in seen or not isinstance(size, int) or isinstance(size, bool) or size <= 0:
            raise ConfigurationError("Release file paths must be unique and sizes must be positive")
        seen.add(file_path)
        files.append(
            ReleaseFile(
                path=file_path,
                sha256=_sha256(item["sha256"], f"files[{index}].sha256"),
                size_bytes=size,
            )
        )
    return ModelRelease(
        repository=repository,
        tag=tag,
        asset=asset,
        asset_sha256=_sha256(release["asset_sha256"], "release.asset_sha256"),
        split_manifest_sha256=_sha256(root["split_manifest_sha256"], "split_manifest_sha256"),
        files=tuple(files),
    )


def model_release_status(manifest_path: Path, root: Path = Path(".")) -> dict[str, object]:
    """Report whether every release file is installed and byte-identical."""
    release = load_model_release(manifest_path)
    root = root.resolve()
    entries: list[dict[str, object]] = []
    ready = True
    for item in release.files:
        target = root / item.path
        if not target.is_file():
            state = "missing"
            actual_sha256 = None
        else:
            actual_sha256 = sha256_file(target)
            state = "ready" if actual_sha256 == item.sha256 else "hash_mismatch"
        ready = ready and state == "ready"
        entries.append(
            {
                "path": item.path.as_posix(),
                "status": state,
                "expected_sha256": item.sha256,
                "actual_sha256": actual_sha256,
                "size_bytes": target.stat().st_size if target.is_file() else None,
            }
        )
    return {
        "status": "ready" if ready else "incomplete",
        "release": release.tag,
        "asset": release.asset,
        "files": entries,
    }


def build_release_archive(
    manifest_path: Path, output_path: Path, root: Path = Path(".")
) -> dict[str, object]:
    """Build a deterministic ZIP asset from verified local training artifacts."""
    release = load_model_release(manifest_path)
    root = root.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.tmp")
    with zipfile.ZipFile(
        temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as archive:
        for item in release.files:
            source = root / item.path
            if not source.is_file():
                raise DataValidationError(f"Missing release source: {item.path.as_posix()}")
            if source.stat().st_size != item.size_bytes or sha256_file(source) != item.sha256:
                raise DataValidationError(f"Release source changed: {item.path.as_posix()}")
            info = zipfile.ZipInfo(item.path.as_posix(), date_time=(2026, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, source.read_bytes(), compresslevel=9)
    temporary.replace(output_path)
    archive_sha256 = sha256_file(output_path)
    if release.asset_sha256 != "0" * 64 and archive_sha256 != release.asset_sha256:
        raise DataValidationError("Packaged release archive differs from the frozen manifest")
    return {
        "status": "complete",
        "path": str(output_path),
        "size_bytes": output_path.stat().st_size,
        "sha256": archive_sha256,
        "files": len(release.files),
    }


def _download(url: str, destination: Path) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": "shopee-demo-artifact-installer"})
    try:
        with (
            urllib.request.urlopen(request, timeout=120) as response,
            destination.open("wb") as out,
        ):
            shutil.copyfileobj(response, out, length=1024 * 1024)
    except OSError as exc:
        raise DataValidationError(f"Cannot download model release from {url}") from exc


def _verify_archive_members(archive: zipfile.ZipFile, release: ModelRelease) -> None:
    expected = {item.path.as_posix() for item in release.files}
    actual: set[str] = set()
    for member in archive.infolist():
        pure = PurePosixPath(member.filename)
        mode = member.external_attr >> 16
        if (
            member.is_dir()
            or pure.is_absolute()
            or ".." in pure.parts
            or "." in pure.parts
            or (mode & 0o170000) == 0o120000
        ):
            raise DataValidationError("Release archive contains an unsafe member")
        actual.add(pure.as_posix())
    if actual != expected:
        raise DataValidationError("Release archive contents differ from the manifest")


def install_model_release(
    manifest_path: Path,
    *,
    root: Path = Path("."),
    url: str | None = None,
    force: bool = False,
) -> dict[str, object]:
    """Download and atomically install a hash-locked inference release."""
    release = load_model_release(manifest_path)
    root = root.resolve()
    status = model_release_status(manifest_path, root)
    if status["status"] == "ready":
        return {**status, "downloaded": False}
    conflicts = [
        entry["path"]
        for entry in cast(list[dict[str, object]], status["files"])
        if entry["status"] == "hash_mismatch"
    ]
    if conflicts and not force:
        joined = ", ".join(str(value) for value in conflicts)
        raise OutputConflictError(f"Refusing to replace changed local artifacts: {joined}")
    staging_parent = root / "artifacts" / ".release_staging"
    staging_parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=staging_parent) as temporary_name:
        temporary_root = Path(temporary_name)
        archive_path = temporary_root / release.asset
        _download(url or release.download_url, archive_path)
        if sha256_file(archive_path) != release.asset_sha256:
            raise DataValidationError("Downloaded release archive failed SHA-256 verification")
        try:
            with zipfile.ZipFile(archive_path) as archive:
                _verify_archive_members(archive, release)
                archive.extractall(temporary_root / "extracted")
        except zipfile.BadZipFile as exc:
            raise DataValidationError(
                "Downloaded release asset is not a valid ZIP archive"
            ) from exc
        extracted = temporary_root / "extracted"
        for item in release.files:
            source = extracted / item.path
            if source.stat().st_size != item.size_bytes or sha256_file(source) != item.sha256:
                raise DataValidationError(
                    f"Extracted release file failed verification: {item.path.as_posix()}"
                )
        for item in release.files:
            source = extracted / item.path
            target = root / item.path
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary_target = target.with_name(f".{target.name}.installing")
            shutil.copyfile(source, temporary_target)
            temporary_target.replace(target)
    final_status = model_release_status(manifest_path, root)
    if final_status["status"] != "ready":
        raise DataValidationError("Installed release did not pass final verification")
    return {**final_status, "downloaded": True, "url": url or release.download_url}


def write_release_summary(result: dict[str, object]) -> str:
    """Stable JSON output shared by CLI commands and tests."""
    return json.dumps(result, indent=2, sort_keys=True)
