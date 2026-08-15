"""Strict versioned configuration for reproducible project commands."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, TypeVar

import yaml

from shopee_match.errors import ConfigurationError

T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class ProjectConfig:
    name: str
    seed: int


@dataclass(frozen=True, slots=True)
class DataConfig:
    metadata_csv: Path
    image_dir: Path
    required_columns: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class OutputConfig:
    online_top_k: int
    match_threshold: float
    review_threshold: float


@dataclass(frozen=True, slots=True)
class LoggingConfig:
    level: str


@dataclass(frozen=True, slots=True)
class AppConfig:
    config_version: str
    project: ProjectConfig
    data: DataConfig
    output: OutputConfig
    logging: LoggingConfig


def _mapping(value: Any, location: str) -> dict[str, Any]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ConfigurationError(f"{location} must be a mapping with string keys")
    return value


def _only_keys(value: dict[str, Any], allowed: set[str], location: str) -> None:
    unknown = set(value) - allowed
    if unknown:
        raise ConfigurationError(f"Unknown keys in {location}: {sorted(unknown)}")
    missing = allowed - set(value)
    if missing:
        raise ConfigurationError(f"Missing keys in {location}: {sorted(missing)}")


def _typed(value: Any, expected: type[T], location: str) -> T:
    if not isinstance(value, expected):
        raise ConfigurationError(f"{location} must be {expected.__name__}")
    return value


def _float(value: Any, location: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ConfigurationError(f"{location} must be numeric")
    return float(value)


def _relative_path(value: Any, location: str) -> Path:
    raw = _typed(value, str, location)
    variants = (PurePosixPath(raw), PureWindowsPath(raw))
    if any(path.is_absolute() or ".." in path.parts for path in variants):
        raise ConfigurationError(f"{location} must be a project-relative path without '..'")
    return Path(raw)


def load_config(path: Path) -> AppConfig:
    """Load a strict YAML config without resolving paths to machine-specific values."""
    try:
        raw: Any = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigurationError(f"Cannot load configuration {path}: {exc}") from exc
    root = _mapping(raw, "config")
    _only_keys(root, {"config_version", "project", "data", "output", "logging"}, "config")

    project = _mapping(root["project"], "project")
    _only_keys(project, {"name", "seed"}, "project")
    seed = _typed(project["seed"], int, "project.seed")
    if isinstance(seed, bool) or not 0 <= seed <= 2**32 - 1:
        raise ConfigurationError("project.seed must be an integer in [0, 2**32 - 1]")

    data = _mapping(root["data"], "data")
    _only_keys(data, {"metadata_csv", "image_dir", "required_columns"}, "data")
    columns = _typed(data["required_columns"], list, "data.required_columns")
    if not columns or not all(isinstance(item, str) and item for item in columns):
        raise ConfigurationError("data.required_columns must be a non-empty list of strings")
    if len(columns) != len(set(columns)):
        raise ConfigurationError("data.required_columns contains duplicates")

    output = _mapping(root["output"], "output")
    _only_keys(
        output,
        {"online_top_k", "match_threshold", "review_threshold"},
        "output",
    )
    top_k = _typed(output["online_top_k"], int, "output.online_top_k")
    match_threshold = _float(output["match_threshold"], "output.match_threshold")
    review_threshold = _float(output["review_threshold"], "output.review_threshold")
    if isinstance(top_k, bool) or top_k <= 0:
        raise ConfigurationError("output.online_top_k must be positive")
    if not 0 <= review_threshold < match_threshold <= 1:
        raise ConfigurationError(
            "Thresholds must satisfy 0 <= review_threshold < match_threshold <= 1"
        )

    logging = _mapping(root["logging"], "logging")
    _only_keys(logging, {"level"}, "logging")
    log_level = _typed(logging["level"], str, "logging.level").upper()
    if log_level not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
        raise ConfigurationError(f"Unsupported logging.level: {log_level}")

    return AppConfig(
        config_version=_typed(root["config_version"], str, "config_version"),
        project=ProjectConfig(_typed(project["name"], str, "project.name"), seed),
        data=DataConfig(
            _relative_path(data["metadata_csv"], "data.metadata_csv"),
            _relative_path(data["image_dir"], "data.image_dir"),
            tuple(columns),
        ),
        output=OutputConfig(top_k, match_threshold, review_threshold),
        logging=LoggingConfig(log_level),
    )
