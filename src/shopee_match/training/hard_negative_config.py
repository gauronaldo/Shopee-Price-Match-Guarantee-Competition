"""Strict Phase 6 hard-negative mining and training configuration."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from shopee_match.errors import ConfigurationError
from shopee_match.training.multimodal_config import (
    MultimodalExperimentConfig,
    load_multimodal_experiment_config,
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
from shopee_match.training.text_evaluation_config import sha256_file


@dataclass(frozen=True, slots=True)
class HardNegativeSourceConfig:
    multimodal_config_path: Path
    multimodal_config_sha256: str
    checkpoint_path: Path
    checkpoint_sha256: str
    metrics_path: Path
    metrics_sha256: str
    experiment: MultimodalExperimentConfig
    metrics: dict[str, Any]


@dataclass(frozen=True, slots=True)
class HardNegativeMiningConfig:
    candidate_k: int
    negatives_per_query: int
    minimum_pair_probability: float
    maximum_pair_probability: float
    similarity_block_size: int
    pair_batch_size: int
    exclude_same_phash: bool
    exclude_exact_normalized_title: bool
    variant_priority_fraction: float


@dataclass(frozen=True, slots=True)
class HardNegativeTrainingConfig:
    device: str
    trainable_components: str
    epochs: int
    products_per_batch: int
    samples_per_product: int
    batches_per_epoch: int
    learning_rate: float
    weight_decay: float
    minimum_learning_rate: float
    gradient_clip_norm: float
    early_stopping_patience: int
    hard_pairs_per_batch: int
    hard_negative_loss_fraction: float
    deterministic: bool


@dataclass(frozen=True, slots=True)
class HardNegativeEvaluationConfig:
    recall_at: tuple[int, ...]
    average_precision_at: int
    candidate_k: int
    checkpoint_metric: str
    checkpoint_target: str
    maximum_map_drop: float
    maximum_recall_at_20_drop: float


@dataclass(frozen=True, slots=True)
class HardNegativeArtifactConfig:
    root: Path
    manifest: Path
    manifest_metadata: Path
    training_root: Path
    checkpoint: Path
    metrics: Path
    report: Path


@dataclass(frozen=True, slots=True)
class HardNegativeExperimentConfig:
    config_version: str
    seed: int
    source: HardNegativeSourceConfig
    mining: HardNegativeMiningConfig
    training: HardNegativeTrainingConfig
    evaluation: HardNegativeEvaluationConfig
    artifacts: HardNegativeArtifactConfig
    config_path: Path


def _sha256_digest(value: Any, location: str) -> str:
    result = _typed(value, str, location).lower()
    if len(result) != 64 or any(character not in "0123456789abcdef" for character in result):
        raise ConfigurationError(f"{location} must be a lowercase SHA-256 hex digest")
    return result


def _fraction(value: Any, location: str, *, positive: bool = False) -> float:
    result = _number(value, location, allow_zero=not positive)
    if result > 1:
        raise ConfigurationError(f"{location} must be inside [0, 1]")
    return result


def _verified_path(mapping: dict[str, Any], name: str) -> tuple[Path, str]:
    path = _relative_path(mapping[name], f"source.{name}")
    expected = _sha256_digest(mapping[f"{name}_sha256"], f"source.{name}_sha256")
    try:
        actual = sha256_file(path)
        if path.suffix.lower() in {".yaml", ".yml"}:
            # Git may materialize YAML with CRLF on Windows. Frozen experiment hashes use
            # canonical LF bytes so the same logical config verifies on every platform.
            canonical = path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
            canonical_actual = hashlib.sha256(canonical).hexdigest()
        else:
            canonical_actual = actual
    except OSError as exc:
        raise ConfigurationError(f"Cannot read frozen Phase 6 source {path}") from exc
    if actual != expected and canonical_actual != expected:
        raise ConfigurationError(
            f"Frozen Phase 6 source hash mismatch for {path}: expected {expected}, got {actual}"
        )
    return path, expected


def load_hard_negative_experiment_config(path: Path) -> HardNegativeExperimentConfig:
    """Load Phase 6 config and verify every Phase 5 dependency before mining."""
    root = _read_yaml(path, "hard-negative experiment config")
    _only_keys(
        root,
        {"config_version", "seed", "source", "mining", "training", "evaluation", "artifacts"},
        "config",
    )
    if root["config_version"] != "phase6.hard_negative_experiment.v1":
        raise ConfigurationError("Unsupported hard-negative config_version")
    seed = _nonnegative_int(root["seed"], "seed")

    source_raw = _mapping(root["source"], "source")
    _only_keys(
        source_raw,
        {
            "multimodal_config",
            "multimodal_config_sha256",
            "checkpoint",
            "checkpoint_sha256",
            "metrics",
            "metrics_sha256",
        },
        "source",
    )
    multimodal_path, multimodal_sha = _verified_path(source_raw, "multimodal_config")
    checkpoint_path, checkpoint_sha = _verified_path(source_raw, "checkpoint")
    metrics_path, metrics_sha = _verified_path(source_raw, "metrics")
    experiment = load_multimodal_experiment_config(multimodal_path)
    try:
        metrics = cast(dict[str, Any], json.loads(metrics_path.read_text(encoding="utf-8")))
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigurationError("Cannot read frozen Phase 5 metrics") from exc
    if checkpoint_path != experiment.artifacts.root / "best.pt":
        raise ConfigurationError("Phase 6 checkpoint is not the Phase 5 best checkpoint")
    selection = cast(dict[str, Any], metrics.get("selection", {}))
    if (
        selection.get("split") != "validation"
        or selection.get("metric") != experiment.evaluation.checkpoint_metric
        or selection.get("target") != experiment.evaluation.checkpoint_target
    ):
        raise ConfigurationError("Phase 5 selection metadata differs from its training config")
    if metrics.get("test", {}).get("status") != "disabled_until_checkpoint_and_protocol_are_frozen":
        raise ConfigurationError(
            "Phase 6 source metrics must be the validation-only training record"
        )
    source = HardNegativeSourceConfig(
        multimodal_config_path=multimodal_path,
        multimodal_config_sha256=multimodal_sha,
        checkpoint_path=checkpoint_path,
        checkpoint_sha256=checkpoint_sha,
        metrics_path=metrics_path,
        metrics_sha256=metrics_sha,
        experiment=experiment,
        metrics=metrics,
    )

    mining_raw = _mapping(root["mining"], "mining")
    _only_keys(
        mining_raw,
        {
            "split",
            "candidate_k",
            "negatives_per_query",
            "minimum_pair_probability",
            "maximum_pair_probability",
            "similarity_block_size",
            "pair_batch_size",
            "exclude_same_phash",
            "exclude_exact_normalized_title",
            "variant_priority_fraction",
        },
        "mining",
    )
    if mining_raw["split"] != "train":
        raise ConfigurationError("Hard negatives may be mined only from train")
    minimum_probability = _fraction(
        mining_raw["minimum_pair_probability"], "mining.minimum_pair_probability"
    )
    maximum_probability = _fraction(
        mining_raw["maximum_pair_probability"],
        "mining.maximum_pair_probability",
        positive=True,
    )
    if minimum_probability >= maximum_probability:
        raise ConfigurationError("Mining probability bounds must be strictly increasing")
    candidate_k = _positive_int(mining_raw["candidate_k"], "mining.candidate_k")
    negatives_per_query = _positive_int(
        mining_raw["negatives_per_query"], "mining.negatives_per_query"
    )
    if candidate_k < negatives_per_query:
        raise ConfigurationError("mining.candidate_k must cover negatives_per_query")
    mining = HardNegativeMiningConfig(
        candidate_k=candidate_k,
        negatives_per_query=negatives_per_query,
        minimum_pair_probability=minimum_probability,
        maximum_pair_probability=maximum_probability,
        similarity_block_size=_positive_int(
            mining_raw["similarity_block_size"], "mining.similarity_block_size"
        ),
        pair_batch_size=_positive_int(mining_raw["pair_batch_size"], "mining.pair_batch_size"),
        exclude_same_phash=_typed(
            mining_raw["exclude_same_phash"], bool, "mining.exclude_same_phash"
        ),
        exclude_exact_normalized_title=_typed(
            mining_raw["exclude_exact_normalized_title"],
            bool,
            "mining.exclude_exact_normalized_title",
        ),
        variant_priority_fraction=_fraction(
            mining_raw["variant_priority_fraction"], "mining.variant_priority_fraction"
        ),
    )

    training_raw = _mapping(root["training"], "training")
    _only_keys(
        training_raw,
        {
            "device",
            "trainable_components",
            "epochs",
            "products_per_batch",
            "samples_per_product",
            "batches_per_epoch",
            "learning_rate",
            "weight_decay",
            "minimum_learning_rate",
            "gradient_clip_norm",
            "early_stopping_patience",
            "hard_pairs_per_batch",
            "hard_negative_loss_fraction",
            "deterministic",
        },
        "training",
    )
    device = _typed(training_raw["device"], str, "training.device")
    if device not in {"auto", "cpu", "cuda"}:
        raise ConfigurationError("training.device must be auto, cpu, or cuda")
    trainable_components = _typed(
        training_raw["trainable_components"], str, "training.trainable_components"
    )
    if trainable_components not in {"fusion_and_pair_head", "pair_head"}:
        raise ConfigurationError(
            "training.trainable_components must be fusion_and_pair_head or pair_head"
        )
    training = HardNegativeTrainingConfig(
        device=device,
        trainable_components=trainable_components,
        epochs=_positive_int(training_raw["epochs"], "training.epochs"),
        products_per_batch=_positive_int(
            training_raw["products_per_batch"], "training.products_per_batch"
        ),
        samples_per_product=_positive_int(
            training_raw["samples_per_product"], "training.samples_per_product"
        ),
        batches_per_epoch=_positive_int(
            training_raw["batches_per_epoch"], "training.batches_per_epoch"
        ),
        learning_rate=_number(training_raw["learning_rate"], "training.learning_rate"),
        weight_decay=_number(
            training_raw["weight_decay"], "training.weight_decay", allow_zero=True
        ),
        minimum_learning_rate=_number(
            training_raw["minimum_learning_rate"],
            "training.minimum_learning_rate",
            allow_zero=True,
        ),
        gradient_clip_norm=_number(
            training_raw["gradient_clip_norm"], "training.gradient_clip_norm"
        ),
        early_stopping_patience=_positive_int(
            training_raw["early_stopping_patience"], "training.early_stopping_patience"
        ),
        hard_pairs_per_batch=_positive_int(
            training_raw["hard_pairs_per_batch"], "training.hard_pairs_per_batch"
        ),
        hard_negative_loss_fraction=_fraction(
            training_raw["hard_negative_loss_fraction"],
            "training.hard_negative_loss_fraction",
            positive=True,
        ),
        deterministic=_typed(training_raw["deterministic"], bool, "training.deterministic"),
    )
    if training.products_per_batch < 2 or training.samples_per_product < 2:
        raise ConfigurationError("Phase 6 product-aware batches require P >= 2 and K >= 2")

    evaluation_raw = _mapping(root["evaluation"], "evaluation")
    _only_keys(
        evaluation_raw,
        {
            "split",
            "evaluate_test",
            "candidate_pool",
            "exclude_query_itself",
            "recall_at",
            "average_precision_at",
            "candidate_k",
            "checkpoint_metric",
            "checkpoint_target",
            "maximum_map_drop",
            "maximum_recall_at_20_drop",
        },
        "evaluation",
    )
    if evaluation_raw["split"] != "validation" or evaluation_raw["evaluate_test"] is not False:
        raise ConfigurationError("Phase 6 may evaluate validation only")
    if (
        evaluation_raw["candidate_pool"] != "full_split"
        or evaluation_raw["exclude_query_itself"] is not True
    ):
        raise ConfigurationError("Phase 6 requires full validation retrieval excluding self")
    recall_values = _typed(evaluation_raw["recall_at"], list, "evaluation.recall_at")
    recall_at = tuple(
        _positive_int(value, f"evaluation.recall_at[{index}]")
        for index, value in enumerate(recall_values)
    )
    ap_at = _positive_int(evaluation_raw["average_precision_at"], "evaluation.average_precision_at")
    validation_candidate_k = _positive_int(evaluation_raw["candidate_k"], "evaluation.candidate_k")
    checkpoint_metric = _typed(
        evaluation_raw["checkpoint_metric"], str, "evaluation.checkpoint_metric"
    )
    checkpoint_target = _typed(
        evaluation_raw["checkpoint_target"], str, "evaluation.checkpoint_target"
    )
    source_evaluation = experiment.evaluation
    if (
        recall_at != source_evaluation.recall_at
        or ap_at != source_evaluation.average_precision_at
        or validation_candidate_k != source_evaluation.candidate_k
        or checkpoint_metric != source_evaluation.checkpoint_metric
        or checkpoint_target != source_evaluation.checkpoint_target
    ):
        raise ConfigurationError("Phase 6 validation protocol must exactly match Phase 5")
    evaluation = HardNegativeEvaluationConfig(
        recall_at=recall_at,
        average_precision_at=ap_at,
        candidate_k=validation_candidate_k,
        checkpoint_metric=checkpoint_metric,
        checkpoint_target=checkpoint_target,
        maximum_map_drop=_number(
            evaluation_raw["maximum_map_drop"], "evaluation.maximum_map_drop", allow_zero=True
        ),
        maximum_recall_at_20_drop=_number(
            evaluation_raw["maximum_recall_at_20_drop"],
            "evaluation.maximum_recall_at_20_drop",
            allow_zero=True,
        ),
    )

    artifact_raw = _mapping(root["artifacts"], "artifacts")
    _only_keys(
        artifact_raw,
        {
            "root",
            "manifest",
            "manifest_metadata",
            "training_root",
            "checkpoint",
            "metrics",
            "report",
        },
        "artifacts",
    )
    artifacts = HardNegativeArtifactConfig(
        root=_relative_path(artifact_raw["root"], "artifacts.root"),
        manifest=_relative_path(artifact_raw["manifest"], "artifacts.manifest"),
        manifest_metadata=_relative_path(
            artifact_raw["manifest_metadata"], "artifacts.manifest_metadata"
        ),
        training_root=_relative_path(artifact_raw["training_root"], "artifacts.training_root"),
        checkpoint=_relative_path(artifact_raw["checkpoint"], "artifacts.checkpoint"),
        metrics=_relative_path(artifact_raw["metrics"], "artifacts.metrics"),
        report=_relative_path(artifact_raw["report"], "artifacts.report"),
    )
    if (
        artifacts.manifest.parent != artifacts.root
        or artifacts.manifest_metadata.parent != artifacts.root
        or artifacts.training_root.parent != artifacts.root
        or artifacts.checkpoint.parent != artifacts.training_root
        or artifacts.metrics.parent != artifacts.training_root
    ):
        raise ConfigurationError("Phase 6 manifest outputs must live directly under artifacts.root")
    return HardNegativeExperimentConfig(
        config_version=str(root["config_version"]),
        seed=seed,
        source=source,
        mining=mining,
        training=training,
        evaluation=evaluation,
        artifacts=artifacts,
        config_path=path,
    )
