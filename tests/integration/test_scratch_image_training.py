from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

from shopee_match.errors import ConfigurationError
from shopee_match.training.image_evaluation_config import load_frozen_image_test_config
from shopee_match.training.image_evaluator import run_frozen_image_test
from shopee_match.training.image_trainer import run_scratch_image_experiment


def _make_image_workspace(root: Path) -> Path:
    images = root / "images"
    images.mkdir()
    rows: list[dict[str, str]] = []
    manifest: list[dict[str, str]] = []
    colors = [
        (220, 30, 30),
        (30, 220, 30),
        (30, 30, 220),
        (180, 180, 30),
        (180, 30, 180),
        (30, 180, 180),
        (100, 60, 220),
        (220, 100, 60),
    ]
    for group_index, color in enumerate(colors):
        split = "train" if group_index < 4 else "validation" if group_index < 6 else "test"
        for variant in range(2):
            posting_id = f"g{group_index}_v{variant}"
            filename = f"{posting_id}.png"
            image = np.full((28 + variant * 2, 34, 3), color, dtype=np.uint8)
            cv2.rectangle(image, (4 + variant, 4), (14 + variant, 20), (255, 255, 255), 2)
            assert cv2.imwrite(str(images / filename), image)
            rows.append(
                {
                    "posting_id": posting_id,
                    "image": filename,
                    "image_phash": f"{group_index:016x}",
                    "title": f"synthetic product {group_index}",
                    "label_group": f"group_{group_index}",
                }
            )
            manifest.append({"posting_id": posting_id, "split": split})
    with (root / "metadata.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (root / "split.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in manifest), encoding="utf-8"
    )
    model = {
        "config_version": "phase3.scratch_image_model.v1",
        "model": {
            "name": "scratch_residual_image_encoder",
            "source": "repository",
            "initialization": "random",
            "pretrained_checkpoint": None,
            "input_channels": 3,
            "stem_width": 4,
            "stage_widths": [4, 8],
            "blocks_per_stage": [1, 1],
            "embedding_dim": 8,
            "projection_hidden_dim": 8,
        },
    }
    (root / "model.yaml").write_text(yaml.safe_dump(model), encoding="utf-8")
    config = {
        "config_version": "phase3.image_embedding_experiment.v1",
        "seed": 2026,
        "data": {
            "metadata_csv": "metadata.csv",
            "split_manifest": "split.jsonl",
            "image_dir": "images",
        },
        "model_config": "model.yaml",
        "preprocessing": {"image_size": 32, "normalization": "fixed_half_range"},
        "training": {
            "device": "cpu",
            "epochs": 2,
            "products_per_batch": 4,
            "samples_per_product": 2,
            "batches_per_epoch": 2,
            "num_workers": 0,
            "learning_rate": 0.002,
            "weight_decay": 0.0001,
            "temperature": 0.1,
            "gradient_clip_norm": 5.0,
            "minimum_learning_rate": 0.00001,
            "early_stopping_patience": 2,
            "deterministic": True,
            "mixed_precision": False,
            "resume_from": None,
        },
        "evaluation": {
            "tune_split": "validation",
            "final_split": "test",
            "evaluate_test": False,
            "candidate_pool": "full_split",
            "exclude_query_itself": True,
            "recall_at": [1, 2, 3],
            "average_precision_at": 3,
            "candidate_k": 3,
            "checkpoint_metric": "map@3",
        },
        "artifacts": {"root": "artifacts/run", "report": "artifacts/run/report.md"},
    }
    (root / "experiment.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return root / "experiment.yaml"


def test_scratch_image_pipeline_runs_to_validation_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = _make_image_workspace(tmp_path)
    monkeypatch.chdir(tmp_path)

    result = run_scratch_image_experiment(config_path.relative_to(tmp_path))
    metrics = json.loads(Path(result["metrics"]).read_text(encoding="utf-8"))

    assert result["status"] == "complete"
    assert result["test_status"].startswith("disabled")
    assert Path(result["checkpoint"]).exists()
    assert Path(result["report"]).exists()

    assert metrics["model"]["parameter_count"] > 0
    assert metrics["validation"]["retrieval"]["queries"] == 4
    assert "group_size" in metrics["validation"]["stratified_retrieval"]
    assert "positive" in metrics["validation"]["similarity_diagnostics"]
    assert Path("artifacts/run/nearest_neighbor_review.json").exists()
    assert metrics["test"]["status"].startswith("disabled")


def test_frozen_checkpoint_evaluation_uses_test_without_reselection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = _make_image_workspace(tmp_path)
    monkeypatch.chdir(tmp_path)
    relative_training_config = config_path.relative_to(tmp_path)
    training_result = run_scratch_image_experiment(relative_training_config)
    checkpoint_path = Path(training_result["checkpoint"])
    training_metrics_path = Path(training_result["metrics"])
    training_metrics = json.loads(training_metrics_path.read_text(encoding="utf-8"))

    def digest(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    frozen_config = {
        "config_version": "phase3.frozen_image_test.v1",
        "seed": 2026,
        "runtime": {"device": "cpu", "num_workers": 0, "batch_size": 4},
        "frozen": {
            "checkpoint": str(checkpoint_path).replace("\\", "/"),
            "checkpoint_sha256": digest(checkpoint_path),
            "training_config": str(relative_training_config).replace("\\", "/"),
            "training_config_sha256": digest(relative_training_config),
            "training_metrics": str(training_metrics_path).replace("\\", "/"),
            "training_metrics_sha256": digest(training_metrics_path),
            "validation_metric": "map@3",
            "validation_metric_value": training_metrics["selection"]["best_metric"],
            "validation_pair_threshold": training_metrics["validation"]["selected_pair_threshold"][
                "threshold"
            ],
        },
        "evaluation": {
            "split": "test",
            "candidate_pool": "full_split",
            "exclude_query_itself": True,
            "recall_at": [1, 2, 3],
            "average_precision_at": 3,
            "candidate_k": 3,
        },
        "artifacts": {
            "root": "artifacts/final",
            "report": "artifacts/final/report.md",
        },
    }
    evaluation_path = Path("frozen_evaluation.yaml")
    evaluation_path.write_text(yaml.safe_dump(frozen_config, sort_keys=False), encoding="utf-8")

    result = run_frozen_image_test(evaluation_path)
    metrics = json.loads(Path(result["metrics"]).read_text(encoding="utf-8"))

    assert result["status"] == "complete"
    assert metrics["test"]["retrieval"]["queries"] == 4
    assert (
        metrics["test"]["pair_at_frozen_validation_threshold"]["threshold"]
        == (frozen_config["frozen"]["validation_pair_threshold"])
    )
    assert Path(result["report"]).exists()

    frozen_config["frozen"]["checkpoint_sha256"] = "0" * 64
    evaluation_path.write_text(yaml.safe_dump(frozen_config, sort_keys=False), encoding="utf-8")
    with pytest.raises(ConfigurationError, match="checkpoint SHA-256 mismatch"):
        load_frozen_image_test_config(evaluation_path)
