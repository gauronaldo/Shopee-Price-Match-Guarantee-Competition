"""Strict configuration for the version-two development protocol."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from shopee_match.errors import ConfigurationError
from shopee_match.hashing import sha256_file
from shopee_match.training.text_config import (
    _mapping,
    _nonnegative_int,
    _only_keys,
    _read_yaml,
    _relative_path,
    _typed,
)


@dataclass(frozen=True, slots=True)
class DevelopmentSourceConfig:
    manifest_path: Path
    manifest_sha256: str
    preserved_split: str


@dataclass(frozen=True, slots=True)
class DevelopmentAllocationConfig:
    seed: int
    train_fraction: float
    development_fraction: float
    confirmation_fraction: float


@dataclass(frozen=True, slots=True)
class DevelopmentArtifactConfig:
    manifest_path: Path
    modeling_manifest_path: Path
    summary_path: Path


@dataclass(frozen=True, slots=True)
class DevelopmentProtocolConfig:
    source: DevelopmentSourceConfig
    allocation: DevelopmentAllocationConfig
    artifacts: DevelopmentArtifactConfig
    config_path: Path


def _digest(value: Any, location: str) -> str:
    result = _typed(value, str, location).lower()
    if len(result) != 64 or any(character not in "0123456789abcdef" for character in result):
        raise ConfigurationError(f"{location} must be a SHA-256 digest")
    return result


def _fraction(value: Any, location: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ConfigurationError(f"{location} must be numeric")
    result = float(value)
    if not 0.0 < result < 1.0:
        raise ConfigurationError(f"{location} must be inside (0, 1)")
    return result


def load_development_protocol_config(path: Path) -> DevelopmentProtocolConfig:
    """Load a protocol that repartitions non-test super-components only."""
    root = _read_yaml(path, "development protocol config")
    _only_keys(root, {"config_version", "source", "allocation", "artifacts"}, "config")
    if root["config_version"] != "entity_resolution.development_protocol.v2":
        raise ConfigurationError("Unsupported development protocol config_version")

    source_raw = _mapping(root["source"], "source")
    _only_keys(source_raw, {"manifest", "manifest_sha256", "preserved_split"}, "source")
    manifest_path = _relative_path(source_raw["manifest"], "source.manifest")
    manifest_sha = _digest(source_raw["manifest_sha256"], "source.manifest_sha256")
    try:
        actual_sha = sha256_file(manifest_path)
    except OSError as exc:
        raise ConfigurationError(f"Cannot read source manifest: {manifest_path}") from exc
    if actual_sha != manifest_sha:
        raise ConfigurationError(
            f"Source manifest mismatch: expected {manifest_sha}, got {actual_sha}"
        )
    preserved = _typed(source_raw["preserved_split"], str, "source.preserved_split")
    if preserved != "test":
        raise ConfigurationError("The historical test split must remain preserved")

    allocation_raw = _mapping(root["allocation"], "allocation")
    allocation_keys = {
        "seed",
        "train_fraction",
        "development_fraction",
        "confirmation_fraction",
    }
    _only_keys(allocation_raw, allocation_keys, "allocation")
    fractions = tuple(
        _fraction(allocation_raw[name], f"allocation.{name}")
        for name in (
            "train_fraction",
            "development_fraction",
            "confirmation_fraction",
        )
    )
    if abs(sum(fractions) - 1.0) > 1e-9:
        raise ConfigurationError("Active development fractions must sum to one")
    allocation = DevelopmentAllocationConfig(
        seed=_nonnegative_int(allocation_raw["seed"], "allocation.seed"),
        train_fraction=fractions[0],
        development_fraction=fractions[1],
        confirmation_fraction=fractions[2],
    )

    artifacts_raw = _mapping(root["artifacts"], "artifacts")
    _only_keys(artifacts_raw, {"manifest", "modeling_manifest", "summary"}, "artifacts")
    artifacts = DevelopmentArtifactConfig(
        manifest_path=_relative_path(artifacts_raw["manifest"], "artifacts.manifest"),
        modeling_manifest_path=_relative_path(
            artifacts_raw["modeling_manifest"], "artifacts.modeling_manifest"
        ),
        summary_path=_relative_path(artifacts_raw["summary"], "artifacts.summary"),
    )
    output_paths = {artifacts.manifest_path, artifacts.modeling_manifest_path}
    if manifest_path in output_paths or len(output_paths) != 2:
        raise ConfigurationError("Development manifests must be distinct from their source")
    return DevelopmentProtocolConfig(
        source=DevelopmentSourceConfig(manifest_path, manifest_sha, preserved),
        allocation=allocation,
        artifacts=artifacts,
        config_path=path,
    )
