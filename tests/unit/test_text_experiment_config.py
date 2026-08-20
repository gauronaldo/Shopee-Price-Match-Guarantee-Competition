from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from shopee_match.errors import ConfigurationError
from shopee_match.training.text_config import load_text_experiment_config


def _write_config(tmp_path: Path, *, pretrained: bool = False, evaluate_test: bool = False) -> Path:
    model = {
        "config_version": "phase4.scratch_text_model.v1",
        "model": {
            "name": "scratch_character_text_cnn",
            "source": "repository",
            "initialization": "random",
            "pretrained_checkpoint": "weights.pt" if pretrained else None,
            "character_embedding_dim": 8,
            "convolution_channels": 8,
            "kernel_sizes": [3, 5],
            "projection_hidden_dim": 8,
            "embedding_dim": 8,
            "dropout": 0.0,
        },
    }
    (tmp_path / "model.yaml").write_text(yaml.safe_dump(model), encoding="utf-8")
    config = {
        "config_version": "phase4.text_embedding_experiment.v1",
        "seed": 2,
        "data": {"metadata_csv": "metadata.csv", "split_manifest": "split.jsonl"},
        "model_config": "model.yaml",
        "tokenization": {
            "level": "character",
            "normalization": "nfkc_casefold_identity_preserving",
            "maximum_length": 32,
            "minimum_frequency": 1,
            "maximum_vocabulary_size": 64,
        },
        "training": {
            "device": "cpu",
            "epochs": 1,
            "products_per_batch": 2,
            "samples_per_product": 2,
            "batches_per_epoch": 1,
            "num_workers": 0,
            "learning_rate": 0.001,
            "weight_decay": 0.0,
            "temperature": 0.07,
            "gradient_clip_norm": 5.0,
            "minimum_learning_rate": 0.0,
            "early_stopping_patience": 1,
            "deterministic": True,
        },
        "evaluation": {
            "tune_split": "validation",
            "final_split": "test",
            "evaluate_test": evaluate_test,
            "candidate_pool": "full_split",
            "exclude_query_itself": True,
            "recall_at": [1, 2],
            "average_precision_at": 2,
            "candidate_k": 2,
            "checkpoint_metric": "map@2",
        },
        "artifacts": {"root": "artifacts/run", "report": "artifacts/run/report.md"},
    }
    path = tmp_path / "experiment.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path


def test_text_config_accepts_random_init_validation_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _write_config(tmp_path)
    monkeypatch.chdir(tmp_path)

    config = load_text_experiment_config(path.relative_to(tmp_path))

    assert config.model_spec.embedding_dim == 8
    assert config.tokenization.maximum_length == 32


def test_text_config_rejects_pretrained_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _write_config(tmp_path, pretrained=True)
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ConfigurationError, match="forbids pretrained"):
        load_text_experiment_config(path.relative_to(tmp_path))


def test_text_config_rejects_test_selection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _write_config(tmp_path, evaluate_test=True)
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ConfigurationError, match="test evaluation disabled"):
        load_text_experiment_config(path.relative_to(tmp_path))
