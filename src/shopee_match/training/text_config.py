"""Strict configuration for the Phase 4 scratch text-embedding experiment."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, TypeVar

import yaml

from shopee_match.errors import ConfigurationError
from shopee_match.models import TextEncoderSpec

T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class TextDataConfig:
    metadata_csv: Path
    split_manifest: Path


@dataclass(frozen=True, slots=True)
class TokenizationConfig:
    maximum_length: int
    minimum_frequency: int
    maximum_vocabulary_size: int


@dataclass(frozen=True, slots=True)
class TextTrainingConfig:
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


@dataclass(frozen=True, slots=True)
class TextEvaluationConfig:
    recall_at: tuple[int, ...]
    average_precision_at: int
    candidate_k: int
    checkpoint_metric: str


@dataclass(frozen=True, slots=True)
class TextArtifactConfig:
    root: Path
    report: Path


@dataclass(frozen=True, slots=True)
class TextExperimentConfig:
    config_version: str
    seed: int
    data: TextDataConfig
    model_config_path: Path
    model_spec: TextEncoderSpec
    tokenization: TokenizationConfig
    training: TextTrainingConfig
    evaluation: TextEvaluationConfig
    artifacts: TextArtifactConfig
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


def _number(value: Any, location: str, *, allow_zero: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ConfigurationError(f"{location} must be numeric")
    result = float(value)
    if result < 0 or (result == 0 and not allow_zero):
        raise ConfigurationError(
            f"{location} must be {'non-negative' if allow_zero else 'positive'}"
        )
    return result


def _relative_path(value: Any, location: str) -> Path:
    raw = _typed(value, str, location)
    variants = (PurePosixPath(raw), PureWindowsPath(raw))
    if any(path.is_absolute() or ".." in path.parts for path in variants):
        raise ConfigurationError(f"{location} must be a project-relative path without '..'")
    return Path(raw)


def _read_yaml(path: Path, location: str) -> dict[str, Any]:
    try:
        raw: Any = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigurationError(f"Cannot load {location} {path}: {exc}") from exc
    return _mapping(raw, location)


def load_text_model_config(path: Path) -> TextEncoderSpec:
    """Load a repository-owned TextCNN configuration and reject pretrained sources."""
    root = _read_yaml(path, "text model config")
    _only_keys(root, {"config_version", "model"}, "text model config")
    if root["config_version"] != "phase4.scratch_text_model.v1":
        raise ConfigurationError("Unsupported scratch text model config_version")
    model = _mapping(root["model"], "model")
    _only_keys(
        model,
        {
            "name",
            "source",
            "initialization",
            "pretrained_checkpoint",
            "character_embedding_dim",
            "convolution_channels",
            "kernel_sizes",
            "projection_hidden_dim",
            "embedding_dim",
            "dropout",
        },
        "model",
    )
    if model["name"] != "scratch_character_text_cnn":
        raise ConfigurationError("Phase 4 supports only scratch_character_text_cnn")
    if model["source"] != "repository" or model["initialization"] != "random":
        raise ConfigurationError("Phase 4 requires repository source and random initialization")
    if model["pretrained_checkpoint"] is not None:
        raise ConfigurationError("Phase 4 forbids pretrained checkpoints")
    kernels = _typed(model["kernel_sizes"], list, "model.kernel_sizes")
    spec = TextEncoderSpec(
        character_embedding_dim=_positive_int(
            model["character_embedding_dim"], "model.character_embedding_dim"
        ),
        convolution_channels=_positive_int(
            model["convolution_channels"], "model.convolution_channels"
        ),
        kernel_sizes=tuple(
            _positive_int(value, f"model.kernel_sizes[{index}]")
            for index, value in enumerate(kernels)
        ),
        projection_hidden_dim=_positive_int(
            model["projection_hidden_dim"], "model.projection_hidden_dim"
        ),
        embedding_dim=_positive_int(model["embedding_dim"], "model.embedding_dim"),
        dropout=_number(model["dropout"], "model.dropout", allow_zero=True),
    )
    try:
        spec.validate()
    except ValueError as exc:
        raise ConfigurationError(str(exc)) from exc
    return spec


def load_text_experiment_config(path: Path) -> TextExperimentConfig:
    """Load a validation-selected text experiment that cannot evaluate test."""
    root = _read_yaml(path, "text experiment config")
    _only_keys(
        root,
        {
            "config_version",
            "seed",
            "data",
            "model_config",
            "tokenization",
            "training",
            "evaluation",
            "artifacts",
        },
        "config",
    )
    if root["config_version"] != "phase4.text_embedding_experiment.v1":
        raise ConfigurationError("Unsupported text experiment config_version")
    seed = _nonnegative_int(root["seed"], "seed")
    if seed > 2**32 - 1:
        raise ConfigurationError("seed must fit in uint32")

    data = _mapping(root["data"], "data")
    _only_keys(data, {"metadata_csv", "split_manifest"}, "data")
    model_path = _relative_path(root["model_config"], "model_config")

    tokenization = _mapping(root["tokenization"], "tokenization")
    _only_keys(
        tokenization,
        {
            "level",
            "normalization",
            "maximum_length",
            "minimum_frequency",
            "maximum_vocabulary_size",
        },
        "tokenization",
    )
    if (
        tokenization["level"] != "character"
        or tokenization["normalization"] != "nfkc_casefold_identity_preserving"
    ):
        raise ConfigurationError(
            "Phase 4 requires character tokenization and identity-preserving normalization"
        )

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
        },
        "training",
    )
    device = _typed(training["device"], str, "training.device")
    if device not in {"auto", "cpu", "cuda"}:
        raise ConfigurationError("training.device must be auto, cpu, or cuda")

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
        raise ConfigurationError("Phase 4 must tune on validation and reserve test")
    if evaluation["evaluate_test"] is not False:
        raise ConfigurationError("Phase 4 training configs must keep test evaluation disabled")
    if (
        evaluation["candidate_pool"] != "full_split"
        or evaluation["exclude_query_itself"] is not True
    ):
        raise ConfigurationError("Text retrieval requires the full split and excludes self")
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
    if checkpoint_metric != f"map@{average_precision_at}":
        raise ConfigurationError(f"checkpoint_metric must be map@{average_precision_at}")

    artifacts = _mapping(root["artifacts"], "artifacts")
    _only_keys(artifacts, {"root", "report"}, "artifacts")
    return TextExperimentConfig(
        config_version=str(root["config_version"]),
        seed=seed,
        data=TextDataConfig(
            _relative_path(data["metadata_csv"], "data.metadata_csv"),
            _relative_path(data["split_manifest"], "data.split_manifest"),
        ),
        model_config_path=model_path,
        model_spec=load_text_model_config(model_path),
        tokenization=TokenizationConfig(
            maximum_length=_positive_int(
                tokenization["maximum_length"], "tokenization.maximum_length"
            ),
            minimum_frequency=_positive_int(
                tokenization["minimum_frequency"], "tokenization.minimum_frequency"
            ),
            maximum_vocabulary_size=_positive_int(
                tokenization["maximum_vocabulary_size"], "tokenization.maximum_vocabulary_size"
            ),
        ),
        training=TextTrainingConfig(
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
            learning_rate=_number(training["learning_rate"], "training.learning_rate"),
            weight_decay=_number(
                training["weight_decay"], "training.weight_decay", allow_zero=True
            ),
            temperature=_number(training["temperature"], "training.temperature"),
            gradient_clip_norm=_number(
                training["gradient_clip_norm"], "training.gradient_clip_norm"
            ),
            minimum_learning_rate=_number(
                training["minimum_learning_rate"], "training.minimum_learning_rate", allow_zero=True
            ),
            early_stopping_patience=_positive_int(
                training["early_stopping_patience"], "training.early_stopping_patience"
            ),
            deterministic=_typed(training["deterministic"], bool, "training.deterministic"),
        ),
        evaluation=TextEvaluationConfig(
            recall_at, average_precision_at, candidate_k, checkpoint_metric
        ),
        artifacts=TextArtifactConfig(
            _relative_path(artifacts["root"], "artifacts.root"),
            _relative_path(artifacts["report"], "artifacts.report"),
        ),
        config_path=path,
    )
