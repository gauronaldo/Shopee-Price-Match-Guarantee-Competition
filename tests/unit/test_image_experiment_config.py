from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from shopee_match.errors import ConfigurationError
from shopee_match.training.image_config import load_image_experiment_config


def _write_configs(
    tmp_path: Path, *, pretrained: bool = False, evaluate_test: bool = False
) -> Path:
    model_path = tmp_path / "model.yaml"
    model = {
        "config_version": "phase3.scratch_image_model.v1",
        "model": {
            "name": "scratch_residual_image_encoder",
            "source": "repository",
            "initialization": "random",
            "pretrained_checkpoint": "weights.pt" if pretrained else None,
            "input_channels": 3,
            "stem_width": 8,
            "stage_widths": [8, 16],
            "blocks_per_stage": [1, 1],
            "embedding_dim": 8,
            "projection_hidden_dim": 16,
        },
    }
    model_path.write_text(yaml.safe_dump(model), encoding="utf-8")
    config = {
        "config_version": "phase3.image_embedding_experiment.v1",
        "seed": 2,
        "data": {
            "metadata_csv": "metadata.csv",
            "split_manifest": "split.jsonl",
            "image_dir": "images",
        },
        "model_config": "model.yaml",
        "preprocessing": {"image_size": 32, "normalization": "fixed_half_range"},
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
            "mixed_precision": False,
            "resume_from": None,
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


def test_image_config_accepts_only_random_init_and_validation_selection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _write_configs(tmp_path)
    monkeypatch.chdir(tmp_path)

    config = load_image_experiment_config(path.relative_to(tmp_path))

    assert config.model_spec.embedding_dim == 8
    assert config.evaluation.evaluate_test is False
    assert config.evaluation.checkpoint_metric == "map@2"


def test_image_config_rejects_pretrained_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _write_configs(tmp_path, pretrained=True)
    monkeypatch.chdir(tmp_path)

    with pytest.raises(ConfigurationError, match="forbids pretrained"):
        load_image_experiment_config(path.relative_to(tmp_path))


def test_image_config_rejects_test_evaluation_during_training(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _write_configs(tmp_path, evaluate_test=True)
    monkeypatch.chdir(tmp_path)

    with pytest.raises(ConfigurationError, match="test evaluation disabled"):
        load_image_experiment_config(path.relative_to(tmp_path))
