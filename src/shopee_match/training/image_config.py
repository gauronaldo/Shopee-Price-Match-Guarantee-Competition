"""Strict configuration for the Phase 3 scratch image-embedding experiment."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, TypeVar

import yaml

from shopee_match.errors import ConfigurationError
from shopee_match.models.image_encoder import ImageEncoderSpec

T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class ImageDataConfig:
    metadata_csv: Path
    split_manifest: Path
    image_dir: Path


@dataclass(frozen=True, slots=True)
class ImageTrainingConfig:
    device: str
    epochs: int
    products_per_batch: int
    samples_per_product: int
    batches_per_epoch: int
    num_workers: int
    learning_rate: float
    weight_decay: float
    temperature: float
    gradient_clip_norm: float
    minimum_learning_rate: float
    early_stopping_patience: int
    deterministic: bool
    mixed_precision: bool
    resume_from: Path | None


@dataclass(frozen=True, slots=True)
class ImageEvaluationConfig:
    tune_split: str
    final_split: str
    evaluate_test: bool
    recall_at: tuple[int, ...]
    average_precision_at: int
    candidate_k: int
    checkpoint_metric: str


@dataclass(frozen=True, slots=True)
class ImageArtifactConfig:
    root: Path
    report: Path


@dataclass(frozen=True, slots=True)
class ImageExperimentConfig:
    config_version: str
    seed: int
    data: ImageDataConfig
    model_config_path: Path
    model_spec: ImageEncoderSpec
    image_size: int
    training: ImageTrainingConfig
    evaluation: ImageEvaluationConfig
    artifacts: ImageArtifactConfig
    config_path: Path


def _mapping(value: Any, location: str) -> dict[str, Any]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ConfigurationError(f"{location} must be a mapping with string keys")
    return value


def _only_keys(value: dict[str, Any], expected: set[str], location: str) -> None:
    missing, unknown = expected - set(value), set(value) - expected
    if missing:
        raise ConfigurationError(f"Missing keys in {location}: {sorted(missing)}")
    if unknown:
        raise ConfigurationError(f"Unknown keys in {location}: {sorted(unknown)}")


def _typed(value: Any, expected: type[T], location: str) -> T:
    if not isinstance(value, expected):
        raise ConfigurationError(f"{location} must be {expected.__name__}")
    return value


def _positive_int(value: Any, location: str) -> int:
    result = _typed(value, int, location)
    if isinstance(result, bool) or result <= 0:
        raise ConfigurationError(f"{location} must be a positive integer")
    return result


def _nonnegative_int(value: Any, location: str) -> int:
    result = _typed(value, int, location)
    if isinstance(result, bool) or result < 0:
        raise ConfigurationError(f"{location} must be a non-negative integer")
    return result


def _positive_float(value: Any, location: str, *, allow_zero: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ConfigurationError(f"{location} must be numeric")
    result = float(value)
    if result < 0 or (result == 0 and not allow_zero):
        qualifier = "non-negative" if allow_zero else "positive"
        raise ConfigurationError(f"{location} must be {qualifier}")
    return result


def _relative_path(value: Any, location: str) -> Path:
    raw = _typed(value, str, location)
    variants = (PurePosixPath(raw), PureWindowsPath(raw))
    if any(path.is_absolute() or ".." in path.parts for path in variants):
        raise ConfigurationError(f"{location} must be a project-relative path without '..'")
    return Path(raw)


def _optional_relative_path(value: Any, location: str) -> Path | None:
    return None if value is None else _relative_path(value, location)


def _read_yaml(path: Path, location: str) -> dict[str, Any]:
    try:
        raw: Any = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigurationError(f"Cannot load {location} {path}: {exc}") from exc
    return _mapping(raw, location)


def load_image_model_config(path: Path) -> ImageEncoderSpec:
    """Load and reject any image model that is not repository-owned and random-init."""
    root = _read_yaml(path, "model config")
    _only_keys(root, {"config_version", "model"}, "model config")
    if root["config_version"] != "phase3.scratch_image_model.v1":
        raise ConfigurationError("Unsupported scratch image model config_version")
    model = _mapping(root["model"], "model")
    _only_keys(
        model,
        {
            "name",
            "source",
            "initialization",
            "pretrained_checkpoint",
            "input_channels",
            "stem_width",
            "stage_widths",
            "blocks_per_stage",
            "embedding_dim",
            "projection_hidden_dim",
        },
        "model",
    )
    if model["name"] != "scratch_residual_image_encoder":
        raise ConfigurationError("Phase 3 supports only scratch_residual_image_encoder")
    if model["source"] != "repository" or model["initialization"] != "random":
        raise ConfigurationError("Phase 3 requires repository source and random initialization")
    if model["pretrained_checkpoint"] is not None:
        raise ConfigurationError("Phase 3 forbids pretrained checkpoints")
    widths_raw = _typed(model["stage_widths"], list, "model.stage_widths")
    blocks_raw = _typed(model["blocks_per_stage"], list, "model.blocks_per_stage")
    spec = ImageEncoderSpec(
        input_channels=_positive_int(model["input_channels"], "model.input_channels"),
        stem_width=_positive_int(model["stem_width"], "model.stem_width"),
        stage_widths=tuple(
            _positive_int(value, f"model.stage_widths[{index}]")
            for index, value in enumerate(widths_raw)
        ),
        blocks_per_stage=tuple(
            _positive_int(value, f"model.blocks_per_stage[{index}]")
            for index, value in enumerate(blocks_raw)
        ),
        embedding_dim=_positive_int(model["embedding_dim"], "model.embedding_dim"),
        projection_hidden_dim=_positive_int(
            model["projection_hidden_dim"], "model.projection_hidden_dim"
        ),
    )
    try:
        spec.validate()
    except ValueError as exc:
        raise ConfigurationError(str(exc)) from exc
    return spec


def load_image_experiment_config(path: Path) -> ImageExperimentConfig:
    """Load an experiment with validation-only checkpoint and threshold selection."""
    root = _read_yaml(path, "image experiment config")
    _only_keys(
        root,
        {
            "config_version",
            "seed",
            "data",
            "model_config",
            "preprocessing",
            "training",
            "evaluation",
            "artifacts",
        },
        "config",
    )
    if root["config_version"] != "phase3.image_embedding_experiment.v1":
        raise ConfigurationError("Unsupported image experiment config_version")
    seed = _nonnegative_int(root["seed"], "seed")
    if seed > 2**32 - 1:
        raise ConfigurationError("seed must fit in uint32")

    data = _mapping(root["data"], "data")
    _only_keys(data, {"metadata_csv", "split_manifest", "image_dir"}, "data")
    model_path = _relative_path(root["model_config"], "model_config")
    preprocessing = _mapping(root["preprocessing"], "preprocessing")
    _only_keys(preprocessing, {"image_size", "normalization"}, "preprocessing")
    if preprocessing["normalization"] != "fixed_half_range":
        raise ConfigurationError("preprocessing.normalization must be fixed_half_range")

    training = _mapping(root["training"], "training")
    _only_keys(
        training,
        {
            "device",
            "epochs",
            "products_per_batch",
            "samples_per_product",
            "batches_per_epoch",
            "num_workers",
            "learning_rate",
            "weight_decay",
            "temperature",
            "gradient_clip_norm",
            "minimum_learning_rate",
            "early_stopping_patience",
            "deterministic",
            "mixed_precision",
            "resume_from",
        },
        "training",
    )
    device = _typed(training["device"], str, "training.device")
    if device not in {"auto", "cpu", "cuda"}:
        raise ConfigurationError("training.device must be auto, cpu, or cuda")
    deterministic = _typed(training["deterministic"], bool, "training.deterministic")
    mixed_precision = _typed(training["mixed_precision"], bool, "training.mixed_precision")
    if mixed_precision:
        raise ConfigurationError("mixed precision remains disabled until float32 gates pass")

    evaluation = _mapping(root["evaluation"], "evaluation")
    _only_keys(
        evaluation,
        {
            "tune_split",
            "final_split",
            "evaluate_test",
            "candidate_pool",
            "exclude_query_itself",
            "recall_at",
            "average_precision_at",
            "candidate_k",
            "checkpoint_metric",
        },
        "evaluation",
    )
    if evaluation["tune_split"] != "validation" or evaluation["final_split"] != "test":
        raise ConfigurationError("Phase 3 must tune on validation and reserve test")
    if evaluation["evaluate_test"] is not False:
        raise ConfigurationError("Phase 3 training configs must keep test evaluation disabled")
    if (
        evaluation["candidate_pool"] != "full_split"
        or evaluation["exclude_query_itself"] is not True
    ):
        raise ConfigurationError("Image retrieval requires the full split and excludes self")
    recall_raw = _typed(evaluation["recall_at"], list, "evaluation.recall_at")
    recall_at = tuple(
        _positive_int(value, f"evaluation.recall_at[{index}]")
        for index, value in enumerate(recall_raw)
    )
    if not recall_at or recall_at != tuple(sorted(set(recall_at))):
        raise ConfigurationError("evaluation.recall_at must be sorted, unique, and non-empty")
    average_precision_at = _positive_int(
        evaluation["average_precision_at"], "evaluation.average_precision_at"
    )
    candidate_k = _positive_int(evaluation["candidate_k"], "evaluation.candidate_k")
    if candidate_k < max(*recall_at, average_precision_at):
        raise ConfigurationError("candidate_k must cover every configured metric K")
    checkpoint_metric = _typed(evaluation["checkpoint_metric"], str, "evaluation.checkpoint_metric")
    expected_metric = f"map@{average_precision_at}"
    if checkpoint_metric != expected_metric:
        raise ConfigurationError(f"checkpoint_metric must be {expected_metric}")

    artifacts = _mapping(root["artifacts"], "artifacts")
    _only_keys(artifacts, {"root", "report"}, "artifacts")
    return ImageExperimentConfig(
        config_version=str(root["config_version"]),
        seed=seed,
        data=ImageDataConfig(
            _relative_path(data["metadata_csv"], "data.metadata_csv"),
            _relative_path(data["split_manifest"], "data.split_manifest"),
            _relative_path(data["image_dir"], "data.image_dir"),
        ),
        model_config_path=model_path,
        model_spec=load_image_model_config(model_path),
        image_size=_positive_int(preprocessing["image_size"], "preprocessing.image_size"),
        training=ImageTrainingConfig(
            device=device,
            epochs=_positive_int(training["epochs"], "training.epochs"),
            products_per_batch=_positive_int(
                training["products_per_batch"], "training.products_per_batch"
            ),
            samples_per_product=_positive_int(
                training["samples_per_product"], "training.samples_per_product"
            ),
            batches_per_epoch=_positive_int(
                training["batches_per_epoch"], "training.batches_per_epoch"
            ),
            num_workers=_nonnegative_int(training["num_workers"], "training.num_workers"),
            learning_rate=_positive_float(training["learning_rate"], "training.learning_rate"),
            weight_decay=_positive_float(
                training["weight_decay"], "training.weight_decay", allow_zero=True
            ),
            temperature=_positive_float(training["temperature"], "training.temperature"),
            gradient_clip_norm=_positive_float(
                training["gradient_clip_norm"], "training.gradient_clip_norm"
            ),
            minimum_learning_rate=_positive_float(
                training["minimum_learning_rate"],
                "training.minimum_learning_rate",
                allow_zero=True,
            ),
            early_stopping_patience=_positive_int(
                training["early_stopping_patience"], "training.early_stopping_patience"
            ),
            deterministic=deterministic,
            mixed_precision=mixed_precision,
            resume_from=_optional_relative_path(training["resume_from"], "training.resume_from"),
        ),
        evaluation=ImageEvaluationConfig(
            tune_split="validation",
            final_split="test",
            evaluate_test=False,
            recall_at=recall_at,
            average_precision_at=average_precision_at,
            candidate_k=candidate_k,
            checkpoint_metric=checkpoint_metric,
        ),
        artifacts=ImageArtifactConfig(
            _relative_path(artifacts["root"], "artifacts.root"),
            _relative_path(artifacts["report"], "artifacts.report"),
        ),
        config_path=path,
    )
