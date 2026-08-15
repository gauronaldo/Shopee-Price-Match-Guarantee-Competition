"""Strict configuration loader for the Phase 1 Shopee data pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, TypeVar

import yaml

from shopee_match.errors import ConfigurationError

T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class DatasetConfig:
    name: str
    competition_slug: str
    metadata_csv: Path
    metadata_sha256: str
    image_dir: Path
    required_columns: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class SplitConfig:
    strategy_version: str
    manifest_path: Path
    summary_path: Path
    seed: int
    train_fraction: float
    validation_fraction: float
    test_fraction: float
    group_key: str
    link_exact_image_reference: bool
    link_exact_sha256: bool
    link_exact_phash: bool


@dataclass(frozen=True, slots=True)
class AuditConfig:
    report_json: Path
    report_markdown: Path
    figure_dir: Path
    inspection_dir: Path
    random_sample_seed: int
    same_group_samples: int
    different_group_samples: int
    near_phash_hamming_distance: int


@dataclass(frozen=True, slots=True)
class Phase1Config:
    config_version: str
    dataset: DatasetConfig
    split: SplitConfig
    audit: AuditConfig
    config_path: Path


def _mapping(value: Any, location: str) -> dict[str, Any]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ConfigurationError(f"{location} must be a mapping with string keys")
    return value


def _typed(value: Any, expected: type[T], location: str) -> T:
    if not isinstance(value, expected):
        raise ConfigurationError(f"{location} must be {expected.__name__}")
    return value


def _only_keys(value: dict[str, Any], expected: set[str], location: str) -> None:
    missing, unknown = expected - set(value), set(value) - expected
    if missing:
        raise ConfigurationError(f"Missing keys in {location}: {sorted(missing)}")
    if unknown:
        raise ConfigurationError(f"Unknown keys in {location}: {sorted(unknown)}")


def _path(value: Any, location: str) -> Path:
    raw = _typed(value, str, location)
    variants = (PurePosixPath(raw), PureWindowsPath(raw))
    if any(item.is_absolute() or ".." in item.parts for item in variants):
        raise ConfigurationError(f"{location} must be a project-relative path without '..'")
    return Path(raw)


def _fraction(value: Any, location: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ConfigurationError(f"{location} must be numeric")
    result = float(value)
    if not 0 < result < 1:
        raise ConfigurationError(f"{location} must be in (0, 1)")
    return result


def load_phase1_config(path: Path) -> Phase1Config:
    """Load and validate a Phase 1 data configuration."""
    try:
        raw: Any = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigurationError(f"Cannot load configuration {path}: {exc}") from exc
    root = _mapping(raw, "config")
    _only_keys(root, {"config_version", "dataset", "split", "audit"}, "config")

    dataset = _mapping(root["dataset"], "dataset")
    dataset_keys = {
        "name",
        "competition_slug",
        "metadata_csv",
        "metadata_sha256",
        "image_dir",
        "required_columns",
    }
    _only_keys(dataset, dataset_keys, "dataset")
    columns = _typed(dataset["required_columns"], list, "dataset.required_columns")
    if not columns or not all(isinstance(item, str) and item for item in columns):
        raise ConfigurationError("dataset.required_columns must be a non-empty string list")
    metadata_sha256 = _typed(dataset["metadata_sha256"], str, "dataset.metadata_sha256").lower()
    if len(metadata_sha256) != 64 or any(
        char not in "0123456789abcdef" for char in metadata_sha256
    ):
        raise ConfigurationError("dataset.metadata_sha256 must be a lowercase SHA-256")

    split = _mapping(root["split"], "split")
    split_keys = {
        "strategy_version",
        "manifest_path",
        "summary_path",
        "seed",
        "train_fraction",
        "validation_fraction",
        "test_fraction",
        "group_key",
        "link_exact_image_reference",
        "link_exact_sha256",
        "link_exact_phash",
    }
    _only_keys(split, split_keys, "split")
    fractions = tuple(
        _fraction(split[key], f"split.{key}")
        for key in ("train_fraction", "validation_fraction", "test_fraction")
    )
    if abs(sum(fractions) - 1) > 1e-9:
        raise ConfigurationError("Split fractions must sum to 1")
    group_key = _typed(split["group_key"], str, "split.group_key")
    if group_key != "label_group":
        raise ConfigurationError("split.group_key must remain label_group")
    seed = _typed(split["seed"], int, "split.seed")
    if isinstance(seed, bool) or not 0 <= seed <= 2**32 - 1:
        raise ConfigurationError("split.seed must be in [0, 2**32 - 1]")

    audit = _mapping(root["audit"], "audit")
    audit_keys = {
        "report_json",
        "report_markdown",
        "figure_dir",
        "inspection_dir",
        "random_sample_seed",
        "same_group_samples",
        "different_group_samples",
        "near_phash_hamming_distance",
    }
    _only_keys(audit, audit_keys, "audit")
    sample_seed = _typed(audit["random_sample_seed"], int, "audit.random_sample_seed")
    same_samples = _typed(audit["same_group_samples"], int, "audit.same_group_samples")
    different_samples = _typed(
        audit["different_group_samples"], int, "audit.different_group_samples"
    )
    near_distance = _typed(
        audit["near_phash_hamming_distance"], int, "audit.near_phash_hamming_distance"
    )
    if min(sample_seed, same_samples, different_samples) < 0 or not 0 <= near_distance <= 8:
        raise ConfigurationError("Audit seeds/counts must be non-negative and pHash distance <= 8")

    split_config = SplitConfig(
        strategy_version=_typed(split["strategy_version"], str, "split.strategy_version"),
        manifest_path=_path(split["manifest_path"], "split.manifest_path"),
        summary_path=_path(split["summary_path"], "split.summary_path"),
        seed=seed,
        train_fraction=fractions[0],
        validation_fraction=fractions[1],
        test_fraction=fractions[2],
        group_key=group_key,
        link_exact_image_reference=_typed(
            split["link_exact_image_reference"], bool, "split.link_exact_image_reference"
        ),
        link_exact_sha256=_typed(split["link_exact_sha256"], bool, "split.link_exact_sha256"),
        link_exact_phash=_typed(split["link_exact_phash"], bool, "split.link_exact_phash"),
    )
    return Phase1Config(
        config_version=_typed(root["config_version"], str, "config_version"),
        dataset=DatasetConfig(
            _typed(dataset["name"], str, "dataset.name"),
            _typed(dataset["competition_slug"], str, "dataset.competition_slug"),
            _path(dataset["metadata_csv"], "dataset.metadata_csv"),
            metadata_sha256,
            _path(dataset["image_dir"], "dataset.image_dir"),
            tuple(columns),
        ),
        split=split_config,
        audit=AuditConfig(
            _path(audit["report_json"], "audit.report_json"),
            _path(audit["report_markdown"], "audit.report_markdown"),
            _path(audit["figure_dir"], "audit.figure_dir"),
            _path(audit["inspection_dir"], "audit.inspection_dir"),
            sample_seed,
            same_samples,
            different_samples,
            near_distance,
        ),
        config_path=path,
    )
