"""Strict configuration for train-only hard-positive pair-head fine-tuning."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from shopee_match.clustering.hybrid_entity_config import (
    HybridEntityConfig,
    load_hybrid_entity_config,
)
from shopee_match.clustering.recovery_config import (
    EntityRecallRecoveryConfig,
    load_entity_recall_recovery_config,
)
from shopee_match.errors import ConfigurationError
from shopee_match.hashing import canonical_text_sha256
from shopee_match.retrieval.config import CandidateRetrievalConfig, load_candidate_retrieval_config
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


@dataclass(frozen=True, slots=True)
class HardPositiveSourceConfig:
    candidate_config_path: Path
    hybrid_entity_config_path: Path
    recovery_config_path: Path
    candidate: CandidateRetrievalConfig
    hybrid_entity: HybridEntityConfig
    recovery: EntityRecallRecoveryConfig


@dataclass(frozen=True, slots=True)
class HardPositiveDataConfig:
    holdout_fraction: float
    split_seed: int


@dataclass(frozen=True, slots=True)
class HardPositiveMiningConfig:
    hard_positive_limit: int
    hard_negative_limit: int
    require_different_phash: bool
    scoring_batch_size: int


@dataclass(frozen=True, slots=True)
class HardPositiveTrainingConfig:
    device: str
    epochs: int
    batches_per_epoch: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    gradient_clip_norm: float
    early_stopping_patience: int
    hard_positive_fraction: float
    hard_negative_fraction: float
    random_positive_fraction: float
    random_negative_fraction: float


@dataclass(frozen=True, slots=True)
class HardPositiveEvaluationConfig:
    holdout_candidate_k: int
    minimum_holdout_precision: float
    minimum_validation_precision: float
    maximum_validation_false_merge_rate: float
    minimum_validation_f1_delta: float


@dataclass(frozen=True, slots=True)
class HardPositiveArtifactConfig:
    root: Path
    checkpoint: Path
    metrics: Path
    report: Path


@dataclass(frozen=True, slots=True)
class HardPositiveExperimentConfig:
    seed: int
    source: HardPositiveSourceConfig
    data: HardPositiveDataConfig
    mining: HardPositiveMiningConfig
    training: HardPositiveTrainingConfig
    evaluation: HardPositiveEvaluationConfig
    artifacts: HardPositiveArtifactConfig
    config_path: Path


def _fraction(value: Any, location: str, *, positive: bool = False) -> float:
    result = _number(value, location, allow_zero=not positive)
    if result > 1.0:
        raise ConfigurationError(f"{location} must be inside [0, 1]")
    return result


def _verified_config(raw: dict[str, Any], name: str) -> Path:
    path = _relative_path(raw[name], f"source.{name}")
    expected = _typed(raw[f"{name}_sha256"], str, f"source.{name}_sha256").lower()
    if len(expected) != 64 or any(character not in "0123456789abcdef" for character in expected):
        raise ConfigurationError(f"source.{name}_sha256 must be a SHA-256 digest")
    try:
        actual = canonical_text_sha256(path)
    except OSError as exc:
        raise ConfigurationError(f"Cannot read hard-positive source config: {path}") from exc
    if actual != expected:
        raise ConfigurationError(
            f"Hard-positive source mismatch for {path}: expected {expected}, got {actual}"
        )
    return path


def load_hard_positive_experiment_config(path: Path) -> HardPositiveExperimentConfig:
    """Load and validate a train-only pair-head fine-tuning experiment."""
    root = _read_yaml(path, "hard-positive experiment config")
    _only_keys(
        root,
        {
            "config_version",
            "seed",
            "source",
            "data",
            "mining",
            "training",
            "evaluation",
            "artifacts",
        },
        "config",
    )
    if root["config_version"] != "pair_head.hard_positive_finetuning.v1":
        raise ConfigurationError("Unsupported hard-positive config_version")
    seed = _nonnegative_int(root["seed"], "seed")

    source_raw = _mapping(root["source"], "source")
    source_names = {"candidate_config", "hybrid_entity_config", "recovery_config"}
    _only_keys(source_raw, source_names | {f"{name}_sha256" for name in source_names}, "source")
    candidate_path = _verified_config(source_raw, "candidate_config")
    hybrid_path = _verified_config(source_raw, "hybrid_entity_config")
    recovery_path = _verified_config(source_raw, "recovery_config")
    candidate = load_candidate_retrieval_config(candidate_path)
    hybrid_entity = load_hybrid_entity_config(hybrid_path)
    recovery = load_entity_recall_recovery_config(recovery_path)
    if (
        seed != candidate.seed
        or seed != hybrid_entity.seed
        or seed != recovery.seed
        or hybrid_entity.source.hybrid.source.phase7_config_path != candidate_path
    ):
        raise ConfigurationError("Hard-positive source experiments are not aligned")
    source = HardPositiveSourceConfig(
        candidate_path,
        hybrid_path,
        recovery_path,
        candidate,
        hybrid_entity,
        recovery,
    )

    data_raw = _mapping(root["data"], "data")
    _only_keys(
        data_raw,
        {"source_split", "holdout_fraction", "split_seed", "evaluate_test"},
        "data",
    )
    if data_raw["source_split"] != "train" or data_raw["evaluate_test"] is not False:
        raise ConfigurationError("Hard-positive training must use train only and disable test")
    data = HardPositiveDataConfig(
        holdout_fraction=_fraction(
            data_raw["holdout_fraction"],
            "data.holdout_fraction",
            positive=True,
        ),
        split_seed=_nonnegative_int(data_raw["split_seed"], "data.split_seed"),
    )

    mining_raw = _mapping(root["mining"], "mining")
    _only_keys(
        mining_raw,
        {
            "hard_positive_limit",
            "hard_negative_limit",
            "require_different_phash",
            "scoring_batch_size",
        },
        "mining",
    )
    mining = HardPositiveMiningConfig(
        hard_positive_limit=_positive_int(
            mining_raw["hard_positive_limit"], "mining.hard_positive_limit"
        ),
        hard_negative_limit=_positive_int(
            mining_raw["hard_negative_limit"], "mining.hard_negative_limit"
        ),
        require_different_phash=_typed(
            mining_raw["require_different_phash"],
            bool,
            "mining.require_different_phash",
        ),
        scoring_batch_size=_positive_int(
            mining_raw["scoring_batch_size"], "mining.scoring_batch_size"
        ),
    )

    training_raw = _mapping(root["training"], "training")
    training_keys = {
        "device",
        "epochs",
        "batches_per_epoch",
        "batch_size",
        "learning_rate",
        "weight_decay",
        "gradient_clip_norm",
        "early_stopping_patience",
        "hard_positive_fraction",
        "hard_negative_fraction",
        "random_positive_fraction",
        "random_negative_fraction",
    }
    _only_keys(training_raw, training_keys, "training")
    device = _typed(training_raw["device"], str, "training.device")
    if device not in {"auto", "cpu", "cuda"}:
        raise ConfigurationError("training.device must be auto, cpu, or cuda")
    fractions = tuple(
        _fraction(training_raw[name], f"training.{name}", positive=True)
        for name in (
            "hard_positive_fraction",
            "hard_negative_fraction",
            "random_positive_fraction",
            "random_negative_fraction",
        )
    )
    if abs(sum(fractions) - 1.0) > 1e-9:
        raise ConfigurationError("training pair fractions must sum to one")
    training = HardPositiveTrainingConfig(
        device=device,
        epochs=_positive_int(training_raw["epochs"], "training.epochs"),
        batches_per_epoch=_positive_int(
            training_raw["batches_per_epoch"], "training.batches_per_epoch"
        ),
        batch_size=_positive_int(training_raw["batch_size"], "training.batch_size"),
        learning_rate=_number(training_raw["learning_rate"], "training.learning_rate"),
        weight_decay=_number(
            training_raw["weight_decay"], "training.weight_decay", allow_zero=True
        ),
        gradient_clip_norm=_number(
            training_raw["gradient_clip_norm"], "training.gradient_clip_norm"
        ),
        early_stopping_patience=_positive_int(
            training_raw["early_stopping_patience"], "training.early_stopping_patience"
        ),
        hard_positive_fraction=fractions[0],
        hard_negative_fraction=fractions[1],
        random_positive_fraction=fractions[2],
        random_negative_fraction=fractions[3],
    )

    evaluation_raw = _mapping(root["evaluation"], "evaluation")
    evaluation_keys = {
        "holdout_candidate_k",
        "minimum_holdout_precision",
        "minimum_validation_precision",
        "maximum_validation_false_merge_rate",
        "minimum_validation_f1_delta",
    }
    _only_keys(evaluation_raw, evaluation_keys, "evaluation")
    evaluation = HardPositiveEvaluationConfig(
        holdout_candidate_k=_positive_int(
            evaluation_raw["holdout_candidate_k"], "evaluation.holdout_candidate_k"
        ),
        minimum_holdout_precision=_fraction(
            evaluation_raw["minimum_holdout_precision"],
            "evaluation.minimum_holdout_precision",
        ),
        minimum_validation_precision=_fraction(
            evaluation_raw["minimum_validation_precision"],
            "evaluation.minimum_validation_precision",
        ),
        maximum_validation_false_merge_rate=_fraction(
            evaluation_raw["maximum_validation_false_merge_rate"],
            "evaluation.maximum_validation_false_merge_rate",
        ),
        minimum_validation_f1_delta=_number(
            evaluation_raw["minimum_validation_f1_delta"],
            "evaluation.minimum_validation_f1_delta",
            allow_zero=True,
        ),
    )

    artifact_raw = _mapping(root["artifacts"], "artifacts")
    _only_keys(artifact_raw, {"root", "checkpoint", "metrics", "report"}, "artifacts")
    artifacts = HardPositiveArtifactConfig(
        root=_relative_path(artifact_raw["root"], "artifacts.root"),
        checkpoint=_relative_path(artifact_raw["checkpoint"], "artifacts.checkpoint"),
        metrics=_relative_path(artifact_raw["metrics"], "artifacts.metrics"),
        report=_relative_path(artifact_raw["report"], "artifacts.report"),
    )
    outputs = (artifacts.checkpoint, artifacts.metrics, artifacts.report)
    if any(output.parent != artifacts.root for output in outputs):
        raise ConfigurationError("Hard-positive outputs must live directly under artifacts.root")
    return HardPositiveExperimentConfig(
        seed,
        source,
        data,
        mining,
        training,
        evaluation,
        artifacts,
        path,
    )
