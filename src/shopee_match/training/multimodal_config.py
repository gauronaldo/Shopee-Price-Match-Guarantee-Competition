"""Strict configuration for Phase 5 scratch multimodal fusion experiments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from shopee_match.errors import ConfigurationError
from shopee_match.models import MultimodalFusionSpec
from shopee_match.training.image_evaluation_config import (
    FrozenImageTestConfig,
    load_frozen_image_test_config,
)
from shopee_match.training.text_config import (
    _mapping,
    _nonnegative_int,
    _number,
    _only_keys,
    _positive_int,
    _read_yaml,
    _relative_path,
    _typed,
)
from shopee_match.training.text_evaluation_config import (
    FrozenTextTestConfig,
    load_frozen_text_test_config,
    sha256_file,
)


@dataclass(frozen=True, slots=True)
class MultimodalDataConfig:
    metadata_csv: Path
    split_manifest: Path
    image_dir: Path


@dataclass(frozen=True, slots=True)
class FrozenEncoderSources:
    image_config: FrozenImageTestConfig
    image_config_sha256: str
    text_config: FrozenTextTestConfig
    text_config_sha256: str


@dataclass(frozen=True, slots=True)
class MultimodalCacheConfig:
    root: Path
    batch_size: int
    num_workers: int


@dataclass(frozen=True, slots=True)
class MultimodalTrainingConfig:
    device: str
    epochs: int
    products_per_batch: int
    samples_per_product: int
    batches_per_epoch: int
    learning_rate: float
    weight_decay: float
    minimum_learning_rate: float
    gradient_clip_norm: float
    early_stopping_patience: int
    deterministic: bool


@dataclass(frozen=True, slots=True)
class MultimodalLossConfig:
    supervised_contrastive_weight: float
    pair_bce_weight: float
    temperature: float
    maximum_negative_ratio: int


@dataclass(frozen=True, slots=True)
class MultimodalEvaluationConfig:
    recall_at: tuple[int, ...]
    average_precision_at: int
    candidate_k: int
    checkpoint_metric: str
    checkpoint_target: str
    simple_fusion_image_weights: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class MultimodalArtifactConfig:
    root: Path
    report: Path


@dataclass(frozen=True, slots=True)
class MultimodalExperimentConfig:
    config_version: str
    seed: int
    data: MultimodalDataConfig
    frozen: FrozenEncoderSources
    cache: MultimodalCacheConfig
    model_config_path: Path
    model_spec: MultimodalFusionSpec
    training: MultimodalTrainingConfig
    loss: MultimodalLossConfig
    evaluation: MultimodalEvaluationConfig
    artifacts: MultimodalArtifactConfig
    config_path: Path


def _sha256_digest(value: Any, location: str) -> str:
    result = _typed(value, str, location).lower()
    if len(result) != 64 or any(character not in "0123456789abcdef" for character in result):
        raise ConfigurationError(f"{location} must be a lowercase SHA-256 hex digest")
    return result


def _verify_frozen_config(path: Path, expected: str, location: str) -> None:
    try:
        actual = sha256_file(path)
    except OSError as exc:
        raise ConfigurationError(f"Cannot read {location}: {path}") from exc
    if actual != expected:
        raise ConfigurationError(f"{location} SHA-256 mismatch: expected {expected}, got {actual}")


def load_multimodal_model_config(path: Path) -> MultimodalFusionSpec:
    root = _read_yaml(path, "multimodal model config")
    _only_keys(root, {"config_version", "model"}, "multimodal model config")
    if root["config_version"] != "phase5.scratch_multimodal_model.v1":
        raise ConfigurationError("Unsupported scratch multimodal model config_version")
    model = _mapping(root["model"], "model")
    required_model_keys = {
        "name",
        "source",
        "initialization",
        "image_embedding_dim",
        "text_embedding_dim",
        "fusion_hidden_dim",
        "joint_embedding_dim",
        "pair_hidden_dim",
        "dropout",
    }
    optional_model_keys = {"fusion_mode", "base_image_weight", "residual_scale"}
    missing = required_model_keys - set(model)
    unknown = set(model) - required_model_keys - optional_model_keys
    if missing:
        raise ConfigurationError(f"Missing keys in model: {sorted(missing)}")
    if unknown:
        raise ConfigurationError(f"Unknown keys in model: {sorted(unknown)}")
    if (
        model["name"] != "learned_multimodal_fusion"
        or model["source"] != "repository"
        or model["initialization"] != "random"
    ):
        raise ConfigurationError("Phase 5 fusion must be repository-owned and randomly initialized")
    spec = MultimodalFusionSpec(
        image_embedding_dim=_positive_int(
            model["image_embedding_dim"], "model.image_embedding_dim"
        ),
        text_embedding_dim=_positive_int(model["text_embedding_dim"], "model.text_embedding_dim"),
        fusion_hidden_dim=_positive_int(model["fusion_hidden_dim"], "model.fusion_hidden_dim"),
        joint_embedding_dim=_positive_int(
            model["joint_embedding_dim"], "model.joint_embedding_dim"
        ),
        pair_hidden_dim=_positive_int(model["pair_hidden_dim"], "model.pair_hidden_dim"),
        dropout=_number(model["dropout"], "model.dropout", allow_zero=True),
        fusion_mode=_typed(model.get("fusion_mode", "projected"), str, "model.fusion_mode"),
        base_image_weight=_number(
            model.get("base_image_weight", 0.5), "model.base_image_weight", allow_zero=True
        ),
        residual_scale=_number(model.get("residual_scale", 1.0), "model.residual_scale"),
    )
    try:
        spec.validate()
    except ValueError as exc:
        raise ConfigurationError(str(exc)) from exc
    return spec


def load_multimodal_experiment_config(path: Path) -> MultimodalExperimentConfig:
    root = _read_yaml(path, "multimodal experiment config")
    _only_keys(
        root,
        {
            "config_version",
            "seed",
            "data",
            "frozen_encoders",
            "cache",
            "model_config",
            "training",
            "loss",
            "evaluation",
            "artifacts",
        },
        "config",
    )
    if root["config_version"] != "phase5.multimodal_experiment.v1":
        raise ConfigurationError("Unsupported multimodal experiment config_version")

    data = _mapping(root["data"], "data")
    _only_keys(data, {"metadata_csv", "split_manifest", "image_dir"}, "data")
    data_config = MultimodalDataConfig(
        metadata_csv=_relative_path(data["metadata_csv"], "data.metadata_csv"),
        split_manifest=_relative_path(data["split_manifest"], "data.split_manifest"),
        image_dir=_relative_path(data["image_dir"], "data.image_dir"),
    )

    frozen = _mapping(root["frozen_encoders"], "frozen_encoders")
    _only_keys(
        frozen,
        {
            "image_evaluation_config",
            "image_evaluation_config_sha256",
            "text_evaluation_config",
            "text_evaluation_config_sha256",
        },
        "frozen_encoders",
    )
    image_path = _relative_path(
        frozen["image_evaluation_config"], "frozen_encoders.image_evaluation_config"
    )
    text_path = _relative_path(
        frozen["text_evaluation_config"], "frozen_encoders.text_evaluation_config"
    )
    image_hash = _sha256_digest(
        frozen["image_evaluation_config_sha256"],
        "frozen_encoders.image_evaluation_config_sha256",
    )
    text_hash = _sha256_digest(
        frozen["text_evaluation_config_sha256"],
        "frozen_encoders.text_evaluation_config_sha256",
    )
    _verify_frozen_config(image_path, image_hash, "frozen image evaluation config")
    _verify_frozen_config(text_path, text_hash, "frozen text evaluation config")
    image_config = load_frozen_image_test_config(image_path)
    text_config = load_frozen_text_test_config(text_path)
    if image_config.training_experiment.data.metadata_csv != data_config.metadata_csv or (
        text_config.training_experiment.data.metadata_csv != data_config.metadata_csv
    ):
        raise ConfigurationError("Frozen encoders and Phase 5 must use the same metadata CSV")
    if image_config.training_experiment.data.split_manifest != data_config.split_manifest or (
        text_config.training_experiment.data.split_manifest != data_config.split_manifest
    ):
        raise ConfigurationError("Frozen encoders and Phase 5 must use the same split manifest")
    if image_config.training_experiment.data.image_dir != data_config.image_dir:
        raise ConfigurationError(
            "Frozen image encoder and Phase 5 must use the same image directory"
        )

    cache = _mapping(root["cache"], "cache")
    _only_keys(cache, {"root", "batch_size", "num_workers"}, "cache")
    cache_config = MultimodalCacheConfig(
        root=_relative_path(cache["root"], "cache.root"),
        batch_size=_positive_int(cache["batch_size"], "cache.batch_size"),
        num_workers=_nonnegative_int(cache["num_workers"], "cache.num_workers"),
    )

    model_path = _relative_path(root["model_config"], "model_config")
    model_spec = load_multimodal_model_config(model_path)
    if model_spec.image_embedding_dim != image_config.training_experiment.model_spec.embedding_dim:
        raise ConfigurationError("Fusion image dimension differs from the frozen image encoder")
    if model_spec.text_embedding_dim != text_config.training_experiment.model_spec.embedding_dim:
        raise ConfigurationError("Fusion text dimension differs from the frozen text encoder")

    training = _mapping(root["training"], "training")
    _only_keys(
        training,
        {
            "device",
            "epochs",
            "products_per_batch",
            "samples_per_product",
            "batches_per_epoch",
            "learning_rate",
            "weight_decay",
            "minimum_learning_rate",
            "gradient_clip_norm",
            "early_stopping_patience",
            "deterministic",
            "freeze_image_encoder",
            "freeze_text_encoder",
        },
        "training",
    )
    device = _typed(training["device"], str, "training.device")
    if device not in {"auto", "cpu", "cuda"}:
        raise ConfigurationError("training.device must be auto, cpu, or cuda")
    if training["freeze_image_encoder"] is not True or training["freeze_text_encoder"] is not True:
        raise ConfigurationError("Initial Phase 5 experiment requires both encoders frozen")
    training_config = MultimodalTrainingConfig(
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
        learning_rate=_number(training["learning_rate"], "training.learning_rate"),
        weight_decay=_number(training["weight_decay"], "training.weight_decay", allow_zero=True),
        minimum_learning_rate=_number(
            training["minimum_learning_rate"], "training.minimum_learning_rate", allow_zero=True
        ),
        gradient_clip_norm=_number(training["gradient_clip_norm"], "training.gradient_clip_norm"),
        early_stopping_patience=_positive_int(
            training["early_stopping_patience"], "training.early_stopping_patience"
        ),
        deterministic=_typed(training["deterministic"], bool, "training.deterministic"),
    )

    loss = _mapping(root["loss"], "loss")
    _only_keys(
        loss,
        {
            "supervised_contrastive_weight",
            "pair_bce_weight",
            "temperature",
            "maximum_negative_ratio",
        },
        "loss",
    )
    loss_config = MultimodalLossConfig(
        supervised_contrastive_weight=_number(
            loss["supervised_contrastive_weight"],
            "loss.supervised_contrastive_weight",
            allow_zero=True,
        ),
        pair_bce_weight=_number(loss["pair_bce_weight"], "loss.pair_bce_weight", allow_zero=True),
        temperature=_number(loss["temperature"], "loss.temperature"),
        maximum_negative_ratio=_positive_int(
            loss["maximum_negative_ratio"], "loss.maximum_negative_ratio"
        ),
    )
    if loss_config.supervised_contrastive_weight + loss_config.pair_bce_weight <= 0:
        raise ConfigurationError("At least one Phase 5 loss weight must be positive")

    evaluation = _mapping(root["evaluation"], "evaluation")
    required_evaluation_keys = {
        "tune_split",
        "final_split",
        "evaluate_test",
        "candidate_pool",
        "exclude_query_itself",
        "recall_at",
        "average_precision_at",
        "candidate_k",
        "checkpoint_metric",
        "simple_fusion_image_weights",
    }
    optional_evaluation_keys = {"checkpoint_target"}
    missing = required_evaluation_keys - set(evaluation)
    unknown = set(evaluation) - required_evaluation_keys - optional_evaluation_keys
    if missing:
        raise ConfigurationError(f"Missing keys in evaluation: {sorted(missing)}")
    if unknown:
        raise ConfigurationError(f"Unknown keys in evaluation: {sorted(unknown)}")
    if (
        evaluation["tune_split"] != "validation"
        or evaluation["final_split"] != "test"
        or evaluation["evaluate_test"] is not False
    ):
        raise ConfigurationError(
            "Phase 5 training must select on validation and keep test disabled"
        )
    if evaluation["candidate_pool"] != "full_split" or (
        evaluation["exclude_query_itself"] is not True
    ):
        raise ConfigurationError("Phase 5 requires full-split retrieval excluding self")
    recall_raw = _typed(evaluation["recall_at"], list, "evaluation.recall_at")
    recall_at = tuple(
        _positive_int(value, f"evaluation.recall_at[{index}]")
        for index, value in enumerate(recall_raw)
    )
    if not recall_at or recall_at != tuple(sorted(set(recall_at))):
        raise ConfigurationError("evaluation.recall_at must be sorted, unique, and non-empty")
    ap_at = _positive_int(evaluation["average_precision_at"], "evaluation.average_precision_at")
    candidate_k = _positive_int(evaluation["candidate_k"], "evaluation.candidate_k")
    if candidate_k < max(*recall_at, ap_at):
        raise ConfigurationError("candidate_k must cover every configured metric K")
    metric = _typed(evaluation["checkpoint_metric"], str, "evaluation.checkpoint_metric")
    if metric != f"map@{ap_at}":
        raise ConfigurationError("checkpoint_metric must equal the configured mAP metric")
    checkpoint_target = _typed(
        evaluation.get("checkpoint_target", "learned_fusion"),
        str,
        "evaluation.checkpoint_target",
    )
    if checkpoint_target not in {"learned_fusion", "pair_head_rerank"}:
        raise ConfigurationError("checkpoint_target must be learned_fusion or pair_head_rerank")
    weight_raw = _typed(
        evaluation["simple_fusion_image_weights"],
        list,
        "evaluation.simple_fusion_image_weights",
    )
    weights = tuple(
        _number(value, f"evaluation.simple_fusion_image_weights[{index}]", allow_zero=True)
        for index, value in enumerate(weight_raw)
    )
    if not weights or weights != tuple(sorted(set(weights))) or any(value > 1 for value in weights):
        raise ConfigurationError("simple fusion weights must be sorted, unique, and inside [0, 1]")
    evaluation_config = MultimodalEvaluationConfig(
        recall_at=recall_at,
        average_precision_at=ap_at,
        candidate_k=candidate_k,
        checkpoint_metric=metric,
        checkpoint_target=checkpoint_target,
        simple_fusion_image_weights=weights,
    )

    artifacts = _mapping(root["artifacts"], "artifacts")
    _only_keys(artifacts, {"root", "report"}, "artifacts")
    return MultimodalExperimentConfig(
        config_version=str(root["config_version"]),
        seed=_nonnegative_int(root["seed"], "seed"),
        data=data_config,
        frozen=FrozenEncoderSources(
            image_config=image_config,
            image_config_sha256=image_hash,
            text_config=text_config,
            text_config_sha256=text_hash,
        ),
        cache=cache_config,
        model_config_path=model_path,
        model_spec=model_spec,
        training=training_config,
        loss=loss_config,
        evaluation=evaluation_config,
        artifacts=MultimodalArtifactConfig(
            root=_relative_path(artifacts["root"], "artifacts.root"),
            report=_relative_path(artifacts["report"], "artifacts.report"),
        ),
        config_path=path,
    )
