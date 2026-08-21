from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest
import yaml

from shopee_match.errors import ConfigurationError
from shopee_match.training.text_evaluator import run_frozen_text_test
from shopee_match.training.text_trainer import (
    refresh_text_training_report,
    run_scratch_text_experiment,
)


def _make_text_workspace(root: Path) -> Path:
    rows: list[dict[str, str]] = []
    manifest: list[dict[str, str]] = []
    for group_index in range(8):
        split = "train" if group_index < 4 else "validation" if group_index < 6 else "test"
        for variant in range(2):
            posting_id = f"g{group_index}_v{variant}"
            rows.append(
                {
                    "posting_id": posting_id,
                    "image": f"unused_{posting_id}.png",
                    "image_phash": f"{group_index:016x}",
                    "title": f"brand product model {group_index} size {variant + 1}00ml",
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
        "config_version": "phase4.scratch_text_model.v1",
        "model": {
            "name": "scratch_character_text_cnn",
            "source": "repository",
            "initialization": "random",
            "pretrained_checkpoint": None,
            "character_embedding_dim": 8,
            "convolution_channels": 8,
            "kernel_sizes": [3, 5],
            "projection_hidden_dim": 8,
            "embedding_dim": 8,
            "dropout": 0.0,
        },
    }
    (root / "model.yaml").write_text(yaml.safe_dump(model), encoding="utf-8")
    config = {
        "config_version": "phase4.text_embedding_experiment.v1",
        "seed": 2026,
        "data": {"metadata_csv": "metadata.csv", "split_manifest": "split.jsonl"},
        "model_config": "model.yaml",
        "tokenization": {
            "level": "character",
            "normalization": "nfkc_casefold_identity_preserving",
            "maximum_length": 48,
            "minimum_frequency": 1,
            "maximum_vocabulary_size": 64,
        },
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
    path = root / "experiment.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path


def test_scratch_text_pipeline_runs_without_test_selection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = _make_text_workspace(tmp_path)
    monkeypatch.chdir(tmp_path)

    result = run_scratch_text_experiment(
        config_path.relative_to(tmp_path), progress_updates_per_epoch=0
    )
    metrics = json.loads(Path(result["metrics"]).read_text(encoding="utf-8"))
    vocabulary = json.loads(Path(result["vocabulary"]).read_text(encoding="utf-8"))

    assert result["status"] == "complete"
    assert result["test_status"].startswith("disabled")
    assert metrics["validation"]["retrieval"]["queries"] == 4
    assert metrics["vocabulary"]["source_split"] == "train"
    assert "normalized_title_length" in metrics["validation"]["stratified_retrieval"]
    assert vocabulary["tokens"][:2] == ["<pad>", "<unk>"]
    assert Path(result["checkpoint"]).exists()
    assert Path(result["report"]).exists()

    metrics["history"] = metrics["history"][:1]
    Path(result["metrics"]).write_text(json.dumps(metrics), encoding="utf-8")
    refreshed = refresh_text_training_report(config_path.relative_to(tmp_path))
    repaired = json.loads(Path(result["metrics"]).read_text(encoding="utf-8"))

    assert refreshed["action"] == "report_refreshed_without_training_or_evaluation"
    assert len(repaired["history"]) == 2
    assert repaired["training_summary"]["completed_epochs"] == 2

    def digest(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    frozen_config = {
        "config_version": "phase4.frozen_text_test.v1",
        "seed": 2026,
        "runtime": {"device": "cpu", "num_workers": 0, "batch_size": 4},
        "frozen": {
            "checkpoint": result["checkpoint"].replace("\\", "/"),
            "checkpoint_sha256": digest(Path(result["checkpoint"])),
            "training_config": str(config_path.relative_to(tmp_path)).replace("\\", "/"),
            "training_config_sha256": digest(config_path.relative_to(tmp_path)),
            "training_metrics": result["metrics"].replace("\\", "/"),
            "training_metrics_sha256": digest(Path(result["metrics"])),
            "validation_metric": "map@3",
            "validation_metric_value": repaired["selection"]["best_metric"],
            "validation_pair_threshold": repaired["validation"]["selected_pair_threshold"][
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
        "artifacts": {"root": "artifacts/final", "report": "artifacts/final/report.md"},
    }
    frozen_path = Path("frozen.yaml")
    frozen_path.write_text(yaml.safe_dump(frozen_config, sort_keys=False), encoding="utf-8")

    final = run_frozen_text_test(frozen_path)

    assert final["status"] == "complete"
    assert Path(final["metrics"]).exists()
    with pytest.raises(ConfigurationError, match="refusing a second"):
        run_frozen_text_test(frozen_path)
