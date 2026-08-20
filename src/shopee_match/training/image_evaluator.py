"""One-time test evaluation for a validation-frozen scratch image checkpoint."""

from __future__ import annotations

import json
import logging
import platform
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from shopee_match.errors import ConfigurationError
from shopee_match.evaluation.image_retrieval import (
    nearest_neighbor_review,
    rank_cosine_embeddings_profiled,
    similarity_diagnostics,
    stratified_retrieval_metrics,
)
from shopee_match.evaluation.protocol import (
    load_splits,
    pair_metrics_at_threshold,
    retrieval_metrics,
)
from shopee_match.models import ScratchResidualImageEncoder
from shopee_match.reproducibility import seed_everything
from shopee_match.training.image_data import ImagePreprocessor, ProductImageDataset
from shopee_match.training.image_evaluation_config import (
    FrozenImageTestConfig,
    load_frozen_image_test_config,
    sha256_file,
)
from shopee_match.training.image_trainer import (
    _format_duration,
    _git_state,
    _resolve_device,
    _write_text_atomic,
    extract_embeddings,
)

LOGGER = logging.getLogger(__name__)


def _checkpoint_spec(config: FrozenImageTestConfig) -> dict[str, Any]:
    spec = config.training_experiment.model_spec
    return {
        "input_channels": spec.input_channels,
        "stem_width": spec.stem_width,
        "stage_widths": spec.stage_widths,
        "blocks_per_stage": spec.blocks_per_stage,
        "embedding_dim": spec.embedding_dim,
        "projection_hidden_dim": spec.projection_hidden_dim,
    }


def _render_report(run: dict[str, Any]) -> str:
    retrieval = run["test"]["retrieval"]
    pair = run["test"]["pair_at_frozen_validation_threshold"]
    latency = run["test"]["search_latency"]
    protocol = run["evaluation_protocol"]
    average_precision_at = protocol["average_precision_at"]
    retrieval_rows = [
        f"| mAP@{average_precision_at} | {retrieval[f'map@{average_precision_at}']:.5f} |"
    ]
    retrieval_rows.extend(
        f"| Recall@{k} | {retrieval[f'recall@{k}']:.5f} |" for k in protocol["recall_at"]
    )
    retrieval_table = "\n".join(retrieval_rows)
    ranking_latency = (
        f"{latency['ranking_p50_ms_per_query']:.3f} / {latency['ranking_p95_ms_per_query']:.3f}"
    )
    return f"""# Scratch image encoder frozen test evaluation

## Protocol

This report evaluates one SHA-256-locked scratch image checkpoint on the held-out test split.
The model, exact Top-{protocol["candidate_k"]} retrieval protocol, and pair threshold were frozen
from validation before test labels were evaluated. No threshold or hyperparameter was selected on
test.

- Checkpoint SHA-256: `{run["frozen_source"]["checkpoint_sha256"]}`
- Training config SHA-256: `{run["frozen_source"]["training_config_sha256"]}`
- Frozen validation mAP@20: `{run["frozen_source"]["validation_metric_value"]:.5f}`
- Frozen validation pair threshold: `{run["frozen_source"]["validation_pair_threshold"]:.6f}`

## Test result

| Metric | Value |
|---|---:|
{retrieval_table}
| Pair precision | {pair["precision"]:.5f} |
| Pair recall | {pair["recall"]:.5f} |
| Pair F1 | {pair["f1"]:.5f} |
| Embedding throughput (listings/s) | {run["test"]["embedding_throughput_per_second"]:.2f} |
| Ranking p50 / p95 (ms/query) | {ranking_latency} |

## Evaluation policy

The frozen checkpoint was evaluated once. The test result was not used to select a checkpoint,
threshold, retrieval setting, or hyperparameter.
"""


def run_frozen_image_test(config_path: Path) -> dict[str, Any]:
    """Evaluate a frozen checkpoint on test without test-time selection."""
    config = load_frozen_image_test_config(config_path)
    seed_everything(config.seed, deterministic=True)
    device = _resolve_device(config.device)
    checkpoint = torch.load(config.checkpoint.path, map_location="cpu", weights_only=False)
    training_metrics = json.loads(
        config.checkpoint.training_metrics_path.read_text(encoding="utf-8")
    )
    if checkpoint.get("checkpoint_version") != "phase3.scratch_image_checkpoint.v1":
        raise ConfigurationError("Unsupported frozen checkpoint version")
    if checkpoint.get("split_manifest_sha256") != sha256_file(
        config.training_experiment.data.split_manifest
    ):
        raise ConfigurationError("Frozen checkpoint split manifest does not match evaluation data")
    if checkpoint.get("model_spec") != _checkpoint_spec(config):
        raise ConfigurationError("Frozen checkpoint architecture differs from training config")
    if checkpoint.get("seed") != config.seed:
        raise ConfigurationError("Frozen checkpoint seed differs from evaluation config")
    if checkpoint.get("checkpoint_metric") != config.checkpoint.validation_metric or not np.isclose(
        float(checkpoint.get("best_metric", float("nan"))),
        config.checkpoint.validation_metric_value,
        rtol=0.0,
        atol=1e-12,
    ):
        raise ConfigurationError("Frozen checkpoint validation selection metadata mismatch")
    recorded_selection = training_metrics.get("selection", {})
    recorded_threshold = training_metrics.get("validation", {}).get("selected_pair_threshold", {})
    if not np.isclose(
        float(recorded_selection.get("best_metric", float("nan"))),
        config.checkpoint.validation_metric_value,
        rtol=0.0,
        atol=1e-12,
    ) or not np.isclose(
        float(recorded_threshold.get("threshold", float("nan"))),
        config.checkpoint.validation_pair_threshold,
        rtol=0.0,
        atol=1e-12,
    ):
        raise ConfigurationError("Frozen training metrics do not match validation locks")

    splits = load_splits(
        config.training_experiment.data.metadata_csv,
        config.training_experiment.data.split_manifest,
    )
    test_split = splits["test"]
    preprocessor = ImagePreprocessor(
        config.training_experiment.image_size, training=False, seed=config.seed
    )
    dataset = ProductImageDataset.for_split(
        test_split, config.training_experiment.data.image_dir, preprocessor
    )
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=device.type == "cuda",
    )
    model = ScratchResidualImageEncoder(config.training_experiment.model_spec)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)

    LOGGER.info(
        "frozen test evaluation: checkpoint=%s device=%s listings=%d",
        config.checkpoint.sha256[:12],
        device,
        len(dataset),
    )
    started = time.perf_counter()
    posting_ids, embeddings, extraction_seconds = extract_embeddings(model, loader, device)
    LOGGER.info("test embeddings extracted; running exact Top-%d ranking", config.candidate_k)
    ranking, search_latency = rank_cosine_embeddings_profiled(
        posting_ids, embeddings, config.candidate_k
    )
    retrieval = retrieval_metrics(
        ranking, test_split.label_by_id, config.recall_at, config.average_precision_at
    )
    pair = pair_metrics_at_threshold(
        ranking, test_split.label_by_id, config.checkpoint.validation_pair_threshold
    )
    commit, dirty = _git_state()
    run: dict[str, Any] = {
        "pipeline_version": "phase3.frozen_image_test.v1",
        "evaluation_provenance": {
            "config_sha256": sha256_file(config.config_path),
            "git_commit": commit,
            "git_dirty": dirty,
            "python": platform.python_version(),
            "torch": torch.__version__,
            "opencv": cv2.__version__,
            "numpy": np.__version__,
            "device": str(device),
            "cuda_device_name": (
                torch.cuda.get_device_name(device) if device.type == "cuda" else None
            ),
            "seed": config.seed,
        },
        "frozen_source": {
            "checkpoint": str(config.checkpoint.path),
            "checkpoint_sha256": config.checkpoint.sha256,
            "training_config": str(config.checkpoint.training_config_path),
            "training_config_sha256": config.checkpoint.training_config_sha256,
            "training_metrics": str(config.checkpoint.training_metrics_path),
            "training_metrics_sha256": config.checkpoint.training_metrics_sha256,
            "training_git_commit": training_metrics["provenance"]["git_commit"],
            "training_git_dirty": training_metrics["provenance"]["git_dirty"],
            "validation_metric": config.checkpoint.validation_metric,
            "validation_metric_value": config.checkpoint.validation_metric_value,
            "validation_pair_threshold": config.checkpoint.validation_pair_threshold,
            "best_epoch": int(checkpoint["best_epoch"]),
        },
        "evaluation_protocol": {
            "split": "test",
            "candidate_pool": "full_split",
            "exclude_query_itself": True,
            "recall_at": config.recall_at,
            "average_precision_at": config.average_precision_at,
            "candidate_k": config.candidate_k,
            "pair_threshold_source": "frozen_validation",
        },
        "test": {
            "queries": len(dataset),
            "retrieval": retrieval,
            "pair_at_frozen_validation_threshold": pair,
            "stratified_retrieval": stratified_retrieval_metrics(
                ranking, test_split, config.recall_at, config.average_precision_at
            ),
            "similarity_diagnostics": similarity_diagnostics(
                posting_ids, embeddings, test_split.label_by_id, seed=config.seed
            ),
            "embedding_extraction_seconds": extraction_seconds,
            "embedding_throughput_per_second": len(dataset) / extraction_seconds,
            "embedding_storage_bytes": embeddings.nbytes,
            "search_latency": search_latency,
            "wall_time_seconds": time.perf_counter() - started,
        },
    }
    metrics_path = config.artifact_root / "metrics.json"
    review_path = config.artifact_root / "nearest_neighbor_review.json"
    _write_text_atomic(metrics_path, json.dumps(run, indent=2, sort_keys=True) + "\n")
    _write_text_atomic(
        review_path,
        json.dumps(nearest_neighbor_review(ranking, test_split), indent=2, sort_keys=True) + "\n",
    )
    _write_text_atomic(config.report_path, _render_report(run))
    maximum_recall_k = max(config.recall_at)
    LOGGER.info(
        "frozen test complete: map@%d=%.5f recall@%d=%.5f elapsed=%s",
        config.average_precision_at,
        retrieval[f"map@{config.average_precision_at}"],
        maximum_recall_k,
        retrieval[f"recall@{maximum_recall_k}"],
        _format_duration(run["test"]["wall_time_seconds"]),
    )
    return {
        "status": "complete",
        "checkpoint_sha256": config.checkpoint.sha256,
        "metrics": str(metrics_path),
        "report": str(config.report_path),
        "test_map": retrieval[f"map@{config.average_precision_at}"],
        "test_recall_at_max": retrieval[f"recall@{maximum_recall_k}"],
    }
