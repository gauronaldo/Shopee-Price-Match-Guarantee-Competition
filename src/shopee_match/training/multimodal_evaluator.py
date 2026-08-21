"""One-time held-out test evaluation for the frozen Phase 5 model."""

from __future__ import annotations

import json
import platform
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from numpy.typing import NDArray

from shopee_match.errors import ConfigurationError
from shopee_match.evaluation.embedding_retrieval import rank_cosine_embeddings_profiled
from shopee_match.evaluation.multimodal_retrieval import (
    rank_simple_score_fusion,
    rerank_with_pair_head,
    unimodal_rankings,
)
from shopee_match.evaluation.protocol import (
    load_splits,
    pair_metrics_at_threshold,
    retrieval_metrics,
)
from shopee_match.models import LearnedMultimodalFusion
from shopee_match.reproducibility import seed_everything
from shopee_match.training.multimodal_data import extract_frozen_multimodal_split
from shopee_match.training.multimodal_evaluation_config import (
    FrozenMultimodalTestConfig,
    load_frozen_multimodal_test_config,
)
from shopee_match.training.multimodal_trainer import (
    _format_duration,
    _git_state,
    _resolve_device,
    _write_text_atomic,
)
from shopee_match.training.text_evaluation_config import sha256_file

FloatArray = NDArray[np.float32]


def ensure_frozen_test_output_absent(config: FrozenMultimodalTestConfig) -> None:
    """Refuse to overwrite evidence from a previous held-out test evaluation."""
    outputs = (
        config.artifact_root / "metrics.json",
        config.report_path,
    )
    existing = [str(path) for path in outputs if path.exists()]
    if existing:
        raise ConfigurationError(
            "Frozen multimodal test has already produced output; refusing to rerun: "
            + ", ".join(existing)
        )


def _joint_embeddings(
    model: LearnedMultimodalFusion,
    image: FloatArray,
    text: FloatArray,
    device: torch.device,
    batch_size: int,
) -> tuple[FloatArray, float]:
    parts: list[FloatArray] = []
    started = time.perf_counter()
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(image), batch_size):
            image_batch = torch.from_numpy(image[start : start + batch_size]).to(device)
            text_batch = torch.from_numpy(text[start : start + batch_size]).to(device)
            parts.append(cast(FloatArray, model(image_batch, text_batch).cpu().numpy()))
    return np.concatenate(parts), time.perf_counter() - started


def _render_report(run: dict[str, Any]) -> str:
    validation = run["validation_reference"]
    test = run["test"]
    rows = "\n".join(
        f"| {label} | {validation[key]['map@20']:.5f} | {test['retrieval'][key]['map@20']:.5f} | "
        f"{validation[key]['recall@20']:.5f} | {test['retrieval'][key]['recall@20']:.5f} |"
        for key, label in (
            ("image_only", "Image only"),
            ("text_only", "Text only"),
            ("simple_score_fusion", "Simple score fusion"),
            ("learned_fusion", "Learned fusion"),
            ("pair_head_rerank", "Pair-head rerank"),
        )
    )
    pair = test["pair_at_frozen_validation_threshold"]
    latency = test["learned_search_latency"]
    ranking_latency = (
        f"{latency['ranking_p50_ms_per_query']:.3f} / "
        f"{latency['ranking_p95_ms_per_query']:.3f} ms/query"
    )
    return f"""# Multimodal fusion frozen test evaluation

## Locked protocol

The canonical seed-2026 checkpoint, training config, metrics, validation-selected pair threshold,
simple-fusion weight, and exact Top-20 protocol were SHA-256 locked before this one-time test run.
No checkpoint, weight, threshold, or hyperparameter was selected on test.

- Checkpoint SHA-256: `{run["frozen_source"]["checkpoint_sha256"]}`
- Training config SHA-256: `{run["frozen_source"]["training_config_sha256"]}`
- Training metrics SHA-256: `{run["frozen_source"]["training_metrics_sha256"]}`
- Frozen pair threshold: `{run["frozen_source"]["validation_pair_threshold"]:.9f}`
- Frozen simple-fusion image weight: `{run["frozen_source"]["simple_fusion_image_weight"]:.2f}`

## Validation-to-test comparison

| Method | Validation mAP@20 | Test mAP@20 | Validation Recall@20 | Test Recall@20 |
|---|---:|---:|---:|---:|
{rows}

## Pair decision at frozen threshold

| Metric | Test value |
|---|---:|
| Precision | {pair["precision"]:.5f} |
| Recall | {pair["recall"]:.5f} |
| F1 | {pair["f1"]:.5f} |

## Efficiency

| Metric | Value |
|---|---:|
| Image extraction throughput | {test["image_extraction_throughput_per_second"]:.2f} listings/s |
| Text extraction throughput | {test["text_extraction_throughput_per_second"]:.2f} listings/s |
| Fusion throughput | {test["fusion_throughput_per_second"]:.2f} listings/s |
| Exact ranking p50 / p95 | {ranking_latency} |

## Interpretation

The pair-head result is the primary learned Phase 5 output because it was the frozen checkpoint
target. mAP@20 measures whether true duplicates are ranked early and completely within Top-20;
Recall@20 measures the candidate ceiling but does not measure false-candidate volume. Pair F1 uses
the validation-frozen decision threshold and therefore assesses the match/no-match operating point
without test-time tuning.
"""


def run_frozen_multimodal_test(config_path: Path) -> dict[str, Any]:
    """Evaluate the locked Phase 5 checkpoint on test exactly once."""
    config = load_frozen_multimodal_test_config(config_path)
    ensure_frozen_test_output_absent(config)
    seed_everything(config.seed, deterministic=True)
    device = _resolve_device(config.device)
    checkpoint = torch.load(config.checkpoint.path, map_location="cpu", weights_only=False)
    training_metrics = cast(
        dict[str, Any],
        json.loads(config.checkpoint.training_metrics_path.read_text(encoding="utf-8")),
    )
    training = config.training_experiment
    if checkpoint.get("checkpoint_version") != "phase5.scratch_multimodal_checkpoint.v1":
        raise ConfigurationError("Unsupported frozen multimodal checkpoint")
    if checkpoint.get("model_spec") != asdict(training.model_spec):
        raise ConfigurationError("Frozen checkpoint architecture differs from training config")
    if checkpoint.get("source_checkpoints") != {
        "image_sha256": training.frozen.image_config.checkpoint.sha256,
        "text_sha256": training.frozen.text_config.checkpoint.sha256,
    }:
        raise ConfigurationError("Frozen checkpoint source encoders differ from training config")
    if checkpoint.get("seed") != config.seed:
        raise ConfigurationError("Frozen checkpoint seed differs from evaluation config")
    if checkpoint.get("checkpoint_metric") != config.checkpoint.validation_metric:
        raise ConfigurationError("Frozen checkpoint metric differs from the validation lock")
    if checkpoint.get("checkpoint_target") != config.checkpoint.checkpoint_target:
        raise ConfigurationError("Frozen checkpoint target differs from the validation lock")
    if not np.isclose(
        float(checkpoint.get("best_metric", float("nan"))),
        config.checkpoint.validation_metric_value,
        rtol=0.0,
        atol=1e-12,
    ):
        raise ConfigurationError("Frozen checkpoint validation metric differs from the lock")
    selection = training_metrics.get("selection", {})
    selected_pair = (
        training_metrics.get("validation", {})
        .get("selected_checkpoint", {})
        .get("pair_head_rerank", {})
    )
    base_fusion = (
        training_metrics.get("validation", {})
        .get("base_ablations", {})
        .get("simple_score_fusion", {})
    )
    if (
        selection.get("target") != config.checkpoint.checkpoint_target
        or not np.isclose(
            float(selection.get("best_metric", float("nan"))),
            config.checkpoint.validation_metric_value,
            rtol=0.0,
            atol=1e-12,
        )
        or not np.isclose(
            float(selected_pair.get("selected_pair_threshold", {}).get("threshold", float("nan"))),
            config.checkpoint.validation_pair_threshold,
            rtol=0.0,
            atol=1e-12,
        )
        or not np.isclose(
            float(base_fusion.get("selected_image_weight", float("nan"))),
            config.checkpoint.simple_fusion_image_weight,
            rtol=0.0,
            atol=1e-12,
        )
    ):
        raise ConfigurationError("Frozen training metrics differ from validation locks")

    splits = load_splits(training.data.metadata_csv, training.data.split_manifest)
    test_split = splits["test"]
    started = time.perf_counter()
    posting_ids, image, text, extraction = extract_frozen_multimodal_split(
        training,
        "test",
        device=device,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
    )
    model = LearnedMultimodalFusion(training.model_spec)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    joint, fusion_seconds = _joint_embeddings(model, image, text, device, config.batch_size)
    image_ranking, text_ranking = unimodal_rankings(posting_ids, image, text, config.candidate_k)
    simple_ranking = rank_simple_score_fusion(
        posting_ids,
        image,
        text,
        image_weight=config.checkpoint.simple_fusion_image_weight,
        candidate_k=config.candidate_k,
    )
    learned_ranking, search_latency = rank_cosine_embeddings_profiled(
        posting_ids, joint, config.candidate_k
    )
    pair_ranking = rerank_with_pair_head(model, posting_ids, joint, learned_ranking, device)
    rankings = {
        "image_only": image_ranking,
        "text_only": text_ranking,
        "simple_score_fusion": simple_ranking,
        "learned_fusion": learned_ranking,
        "pair_head_rerank": pair_ranking,
    }
    test_retrieval = {
        name: retrieval_metrics(
            ranking,
            test_split.label_by_id,
            config.recall_at,
            config.average_precision_at,
        )
        for name, ranking in rankings.items()
    }
    base_validation = training_metrics["validation"]["base_ablations"]
    selected_validation = training_metrics["validation"]["selected_checkpoint"]
    validation_reference = {
        "image_only": base_validation["image_only"],
        "text_only": base_validation["text_only"],
        "simple_score_fusion": base_validation["simple_score_fusion"]["retrieval"],
        "learned_fusion": selected_validation["learned_fusion"]["retrieval"],
        "pair_head_rerank": selected_validation["pair_head_rerank"]["retrieval"],
    }
    commit, dirty = _git_state()
    image_seconds = extraction["image_extraction_seconds"]
    text_seconds = extraction["text_extraction_seconds"]
    run: dict[str, Any] = {
        "pipeline_version": "phase5.frozen_multimodal_test.v1",
        "evaluation_provenance": {
            "config_sha256": sha256_file(config.config_path),
            "git_commit": commit,
            "git_dirty": dirty,
            "seed": config.seed,
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "device": str(device),
            "cuda_device_name": (
                torch.cuda.get_device_name(device) if device.type == "cuda" else None
            ),
        },
        "frozen_source": {
            "checkpoint": str(config.checkpoint.path),
            "checkpoint_sha256": config.checkpoint.sha256,
            "training_config": str(config.checkpoint.training_config_path),
            "training_config_sha256": config.checkpoint.training_config_sha256,
            "training_metrics": str(config.checkpoint.training_metrics_path),
            "training_metrics_sha256": config.checkpoint.training_metrics_sha256,
            "validation_metric": config.checkpoint.validation_metric,
            "validation_metric_value": config.checkpoint.validation_metric_value,
            "checkpoint_target": config.checkpoint.checkpoint_target,
            "validation_pair_threshold": config.checkpoint.validation_pair_threshold,
            "simple_fusion_image_weight": config.checkpoint.simple_fusion_image_weight,
        },
        "evaluation_protocol": {
            "split": "test",
            "candidate_pool": "full_split",
            "exclude_query_itself": True,
            "recall_at": config.recall_at,
            "average_precision_at": config.average_precision_at,
            "candidate_k": config.candidate_k,
            "selection_on_test": False,
        },
        "validation_reference": validation_reference,
        "test": {
            "queries": len(posting_ids),
            "retrieval": test_retrieval,
            "pair_at_frozen_validation_threshold": pair_metrics_at_threshold(
                pair_ranking,
                test_split.label_by_id,
                config.checkpoint.validation_pair_threshold,
            ),
            "image_extraction_seconds": image_seconds,
            "text_extraction_seconds": text_seconds,
            "fusion_seconds": fusion_seconds,
            "image_extraction_throughput_per_second": len(posting_ids) / image_seconds,
            "text_extraction_throughput_per_second": len(posting_ids) / text_seconds,
            "fusion_throughput_per_second": len(posting_ids) / fusion_seconds,
            "embedding_storage_bytes": image.nbytes + text.nbytes + joint.nbytes,
            "learned_search_latency": search_latency,
            "wall_time_seconds": time.perf_counter() - started,
        },
    }
    metrics_path = config.artifact_root / "metrics.json"
    _write_text_atomic(metrics_path, json.dumps(run, indent=2, sort_keys=True) + "\n")
    _write_text_atomic(config.report_path, _render_report(run))
    pair_test = test_retrieval["pair_head_rerank"]
    return {
        "status": "complete",
        "checkpoint_sha256": config.checkpoint.sha256,
        "metrics": str(metrics_path),
        "report": str(config.report_path),
        "test_map_at_20": pair_test["map@20"],
        "test_recall_at_20": pair_test["recall@20"],
        "wall_time": _format_duration(run["test"]["wall_time_seconds"]),
    }
