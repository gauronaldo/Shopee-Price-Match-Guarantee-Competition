"""Strict train-only/validation-only configuration for residual pair evidence."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from shopee_match.clustering.recovery_config import (
    EntityRecallRecoveryConfig,
    load_entity_recall_recovery_config,
)
from shopee_match.errors import ConfigurationError
from shopee_match.hashing import canonical_text_sha256
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
class PairEvidenceTrainingConfig:
    device: str
    epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    early_stopping_patience: int
    hard_positive_fraction: float
    hard_positive_weight: float
    hard_negative_weight: float
    negative_to_positive_ratio: float
    deterministic: bool


@dataclass(frozen=True, slots=True)
class PairEvidenceTfidfConfig:
    ngram_range: tuple[int, int]
    max_features: int


@dataclass(frozen=True, slots=True)
class PairEvidenceArtifactConfig:
    root: Path
    checkpoint: Path
    rescored_pairs: Path
    assignments: Path
    metrics: Path
    review: Path
    report: Path


@dataclass(frozen=True, slots=True)
class PairEvidenceExperimentConfig:
    seed: int
    recovery_config_path: Path
    recovery_config_sha256: str
    recovery: EntityRecallRecoveryConfig
    training: PairEvidenceTrainingConfig
    tfidf: PairEvidenceTfidfConfig
    artifacts: PairEvidenceArtifactConfig
    config_path: Path


def _fraction(value: Any, location: str, *, allow_zero: bool = True) -> float:
    result = _number(value, location, allow_zero=allow_zero)
    if result > 1:
        raise ConfigurationError(f"{location} must be inside [0, 1]")
    return result


def _positive_number(value: Any, location: str) -> float:
    return _number(value, location, allow_zero=False)


def _digest(value: Any, location: str) -> str:
    result = _typed(value, str, location).lower()
    if len(result) != 64 or any(character not in "0123456789abcdef" for character in result):
        raise ConfigurationError(f"{location} must be a SHA-256 digest")
    return result


def load_pair_evidence_experiment_config(path: Path) -> PairEvidenceExperimentConfig:
    """Load the frozen baseline, train-only features, and validation-only selection contract."""
    root = _read_yaml(path, "pair-evidence config")
    _only_keys(
        root,
        {"config_version", "seed", "source", "data", "training", "tfidf", "artifacts"},
        "config",
    )
    if root["config_version"] != "pair_evidence.training.v1":
        raise ConfigurationError("Unsupported pair-evidence config_version")
    seed = _nonnegative_int(root["seed"], "seed")

    source_raw = _mapping(root["source"], "source")
    _only_keys(source_raw, {"recovery_config", "recovery_config_sha256"}, "source")
    recovery_path = _relative_path(source_raw["recovery_config"], "source.recovery_config")
    recovery_sha = _digest(source_raw["recovery_config_sha256"], "source.recovery_config_sha256")
    try:
        actual_sha = canonical_text_sha256(recovery_path)
    except OSError as exc:
        raise ConfigurationError("Cannot read frozen recall-recovery config") from exc
    if actual_sha != recovery_sha:
        raise ConfigurationError(
            "Frozen recall-recovery config hash mismatch: "
            f"expected {recovery_sha}, got {actual_sha}"
        )
    recovery = load_entity_recall_recovery_config(recovery_path)
    if recovery.seed != seed:
        raise ConfigurationError("Pair-evidence and recovery seeds must match")

    data_raw = _mapping(root["data"], "data")
    _only_keys(data_raw, {"fit_split", "tune_split", "evaluate_test"}, "data")
    if (
        data_raw["fit_split"] != "train"
        or data_raw["tune_split"] != "validation"
        or data_raw["evaluate_test"] is not False
    ):
        raise ConfigurationError("Pair evidence must fit on train and tune on validation only")

    training_raw = _mapping(root["training"], "training")
    _only_keys(
        training_raw,
        {
            "device",
            "epochs",
            "batch_size",
            "learning_rate",
            "weight_decay",
            "early_stopping_patience",
            "hard_positive_fraction",
            "hard_positive_weight",
            "hard_negative_weight",
            "negative_to_positive_ratio",
            "deterministic",
        },
        "training",
    )
    device = _typed(training_raw["device"], str, "training.device")
    if device not in {"auto", "cpu", "cuda"}:
        raise ConfigurationError("training.device must be auto, cpu, or cuda")
    training = PairEvidenceTrainingConfig(
        device=device,
        epochs=_positive_int(training_raw["epochs"], "training.epochs"),
        batch_size=_positive_int(training_raw["batch_size"], "training.batch_size"),
        learning_rate=_positive_number(training_raw["learning_rate"], "training.learning_rate"),
        weight_decay=_number(
            training_raw["weight_decay"], "training.weight_decay", allow_zero=True
        ),
        early_stopping_patience=_positive_int(
            training_raw["early_stopping_patience"], "training.early_stopping_patience"
        ),
        hard_positive_fraction=_fraction(
            training_raw["hard_positive_fraction"], "training.hard_positive_fraction"
        ),
        hard_positive_weight=_positive_number(
            training_raw["hard_positive_weight"], "training.hard_positive_weight"
        ),
        hard_negative_weight=_positive_number(
            training_raw["hard_negative_weight"], "training.hard_negative_weight"
        ),
        negative_to_positive_ratio=_positive_number(
            training_raw["negative_to_positive_ratio"], "training.negative_to_positive_ratio"
        ),
        deterministic=_typed(training_raw["deterministic"], bool, "training.deterministic"),
    )

    tfidf_raw = _mapping(root["tfidf"], "tfidf")
    _only_keys(tfidf_raw, {"ngram_range", "max_features"}, "tfidf")
    ngram_raw = _typed(tfidf_raw["ngram_range"], list, "tfidf.ngram_range")
    if len(ngram_raw) != 2:
        raise ConfigurationError("tfidf.ngram_range must contain two integers")
    ngram_range = (
        _positive_int(ngram_raw[0], "tfidf.ngram_range[0]"),
        _positive_int(ngram_raw[1], "tfidf.ngram_range[1]"),
    )
    if ngram_range[0] > ngram_range[1]:
        raise ConfigurationError("tfidf.ngram_range must be increasing")
    tfidf = PairEvidenceTfidfConfig(
        ngram_range=ngram_range,
        max_features=_positive_int(tfidf_raw["max_features"], "tfidf.max_features"),
    )

    artifact_raw = _mapping(root["artifacts"], "artifacts")
    artifact_names = {
        "root",
        "checkpoint",
        "rescored_pairs",
        "assignments",
        "metrics",
        "review",
        "report",
    }
    _only_keys(artifact_raw, artifact_names, "artifacts")
    artifacts = PairEvidenceArtifactConfig(
        root=_relative_path(artifact_raw["root"], "artifacts.root"),
        checkpoint=_relative_path(artifact_raw["checkpoint"], "artifacts.checkpoint"),
        rescored_pairs=_relative_path(artifact_raw["rescored_pairs"], "artifacts.rescored_pairs"),
        assignments=_relative_path(artifact_raw["assignments"], "artifacts.assignments"),
        metrics=_relative_path(artifact_raw["metrics"], "artifacts.metrics"),
        review=_relative_path(artifact_raw["review"], "artifacts.review"),
        report=_relative_path(artifact_raw["report"], "artifacts.report"),
    )
    if any(
        output.parent != artifacts.root
        for output in (
            artifacts.checkpoint,
            artifacts.rescored_pairs,
            artifacts.assignments,
            artifacts.metrics,
            artifacts.review,
            artifacts.report,
        )
    ):
        raise ConfigurationError("Pair-evidence outputs must live directly under artifacts.root")
    return PairEvidenceExperimentConfig(
        seed,
        recovery_path,
        recovery_sha,
        recovery,
        training,
        tfidf,
        artifacts,
        path,
    )
