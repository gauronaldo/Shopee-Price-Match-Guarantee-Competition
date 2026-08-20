"""Configuration-driven Phase 3 training, checkpointing, and validation retrieval."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import platform
import subprocess
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import cv2
import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor, nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from shopee_match.errors import ConfigurationError, DataValidationError
from shopee_match.evaluation.image_retrieval import (
    nearest_neighbor_review,
    rank_cosine_embeddings_profiled,
    similarity_diagnostics,
    stratified_retrieval_metrics,
)
from shopee_match.evaluation.protocol import (
    EvaluationSplit,
    Ranking,
    load_splits,
    retrieval_metrics,
    select_threshold,
)
from shopee_match.losses import SupervisedContrastiveLoss
from shopee_match.models import ScratchResidualImageEncoder
from shopee_match.reproducibility import seed_everything
from shopee_match.training.image_config import (
    ImageExperimentConfig,
    load_image_experiment_config,
)
from shopee_match.training.image_data import (
    ImagePreprocessor,
    ProductBatchSampler,
    ProductImageDataset,
)

LOGGER = logging.getLogger(__name__)
FloatArray = NDArray[np.floating[Any]]


def _format_duration(seconds: float) -> str:
    """Format a progress duration compactly without terminal-control sequences."""
    rounded = max(0, round(seconds))
    hours, remainder = divmod(rounded, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes:02d}m"
    if minutes:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"


def _progress_milestones(total_batches: int, updates: int) -> frozenset[int]:
    """Return bounded batch milestones, including the final batch when enabled."""
    if total_batches <= 0:
        raise ValueError("total_batches must be positive")
    if updates < 0:
        raise ValueError("updates must be non-negative")
    if updates == 0:
        return frozenset()
    return frozenset(
        min(total_batches, max(1, round(total_batches * step / updates)))
        for step in range(1, updates + 1)
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_state() -> tuple[str, bool]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
    except (OSError, subprocess.CalledProcessError):
        return "unavailable", True
    return commit, dirty


def _write_text_atomic(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def _save_checkpoint_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _resolve_device(requested: str) -> torch.device:
    if requested == "cuda" and not torch.cuda.is_available():
        raise ConfigurationError("training.device=cuda but CUDA is not available")
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def _tensor_batch(batch: Mapping[str, Any], key: str, device: torch.device) -> Tensor:
    value = batch[key]
    if not isinstance(value, Tensor):
        raise DataValidationError(f"Batch field {key!r} is not a tensor")
    return value.to(device, non_blocking=device.type == "cuda")


def _posting_ids(batch: Mapping[str, Any]) -> tuple[str, ...]:
    value = batch["posting_id"]
    if not isinstance(value, list | tuple) or not all(isinstance(item, str) for item in value):
        raise DataValidationError("Batch posting_id field is invalid")
    return tuple(value)


def _verify_finite_gradients(model: nn.Module) -> None:
    missing = []
    non_finite = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.grad is None:
            missing.append(name)
        elif not torch.isfinite(parameter.grad).all():
            non_finite.append(name)
    if missing or non_finite:
        raise RuntimeError(
            f"Gradient gate failed; missing={missing[:5]}, non_finite={non_finite[:5]}"
        )


def extract_embeddings(
    model: ScratchResidualImageEncoder,
    loader: DataLoader[dict[str, Tensor | str]],
    device: torch.device,
) -> tuple[tuple[str, ...], FloatArray, float]:
    """Extract embeddings in deterministic dataset order and report wall time."""
    model.eval()
    posting_ids: list[str] = []
    parts: list[FloatArray] = []
    started = time.perf_counter()
    with torch.inference_mode():
        for raw_batch in loader:
            batch = cast(Mapping[str, Any], raw_batch)
            images = _tensor_batch(batch, "image", device)
            parts.append(model(images).cpu().numpy())
            posting_ids.extend(_posting_ids(batch))
    elapsed = time.perf_counter() - started
    if not parts:
        raise DataValidationError("Cannot extract embeddings from an empty split")
    return tuple(posting_ids), np.concatenate(parts, axis=0), elapsed


def _validation_result(
    model: ScratchResidualImageEncoder,
    loader: DataLoader[dict[str, Tensor | str]],
    split: EvaluationSplit,
    config: ImageExperimentConfig,
    device: torch.device,
) -> tuple[dict[str, float], dict[str, float], float, FloatArray, Ranking, dict[str, float]]:
    LOGGER.info("validation: extracting embeddings")
    posting_ids, embeddings, extraction_seconds = extract_embeddings(model, loader, device)
    LOGGER.info("validation: ranking %d listings and computing retrieval metrics", len(posting_ids))
    ranking, search_latency = rank_cosine_embeddings_profiled(
        posting_ids, embeddings, config.evaluation.candidate_k
    )
    metrics = retrieval_metrics(
        ranking,
        split.label_by_id,
        config.evaluation.recall_at,
        config.evaluation.average_precision_at,
    )
    threshold = select_threshold(ranking, split.label_by_id)
    return metrics, threshold, extraction_seconds, embeddings, ranking, search_latency


def _provenance(config: ImageExperimentConfig, device: torch.device) -> dict[str, Any]:
    commit, dirty = _git_state()
    cuda_name = torch.cuda.get_device_name(device) if device.type == "cuda" else None
    return {
        "config_version": config.config_version,
        "config_sha256": _sha256(config.config_path),
        "model_config_sha256": _sha256(config.model_config_path),
        "manifest_sha256": _sha256(config.data.split_manifest),
        "metadata_sha256": _sha256(config.data.metadata_csv),
        "git_commit": commit,
        "git_dirty": dirty,
        "seed": config.seed,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "opencv": cv2.__version__,
        "numpy": np.__version__,
        "platform": platform.platform(),
        "device": str(device),
        "cuda_device_name": cuda_name,
    }


def _checkpoint_payload(
    *,
    config: ImageExperimentConfig,
    model: ScratchResidualImageEncoder,
    optimizer: AdamW,
    scheduler: CosineAnnealingLR,
    epoch: int,
    best_metric: float,
    best_epoch: int,
    history: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "checkpoint_version": "phase3.scratch_image_checkpoint.v1",
        "epoch": epoch,
        "best_metric": best_metric,
        "best_epoch": best_epoch,
        "checkpoint_metric": config.evaluation.checkpoint_metric,
        "model_spec": {
            "input_channels": config.model_spec.input_channels,
            "stem_width": config.model_spec.stem_width,
            "stage_widths": config.model_spec.stage_widths,
            "blocks_per_stage": config.model_spec.blocks_per_stage,
            "embedding_dim": config.model_spec.embedding_dim,
            "projection_hidden_dim": config.model_spec.projection_hidden_dim,
        },
        "initialization_policy": model.initialization_policy,
        "normalization_policy": ImagePreprocessor.normalization_policy,
        "seed": config.seed,
        "split_manifest_sha256": _sha256(config.data.split_manifest),
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),  # type: ignore[no-untyped-call]
        "history": history,
    }


def _load_resume(
    path: Path,
    config: ImageExperimentConfig,
    model: ScratchResidualImageEncoder,
    optimizer: AdamW,
    scheduler: CosineAnnealingLR,
) -> tuple[int, float, int, list[dict[str, Any]]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("checkpoint_version") != "phase3.scratch_image_checkpoint.v1":
        raise ConfigurationError("Unsupported resume checkpoint")
    if payload.get("split_manifest_sha256") != _sha256(config.data.split_manifest):
        raise ConfigurationError("Resume checkpoint was trained on another split manifest")
    model.load_state_dict(payload["model_state"])
    optimizer.load_state_dict(payload["optimizer_state"])
    scheduler.load_state_dict(payload["scheduler_state"])
    history = cast(list[dict[str, Any]], payload.get("history", []))
    return (
        int(payload["epoch"]) + 1,
        float(payload["best_metric"]),
        int(payload["best_epoch"]),
        history,
    )


def _render_report(run: dict[str, Any]) -> str:
    final = run["validation"]
    history_rows = "\n".join(
        f"| {row['epoch']} | {row['train_loss']:.5f} | "
        f"{row['validation'][run['selection']['metric']]:.5f} | {row['epoch_seconds']:.2f} |"
        for row in run["history"]
    )
    metrics = final["retrieval"]
    return f"""# Scratch image encoder benchmark

## Experiment status

This is an image-only Phase 3 run. The residual CNN and projection head were initialized randomly;
no pretrained weights, title features, pHash, or ORB scores entered the model. Checkpoint selection
used validation `{run["selection"]["metric"]}` only. Test evaluation remains disabled until the
configuration and checkpoint are frozen.

## Validation result

- Selected epoch: `{run["selection"]["best_epoch"]}`
- mAP@20: `{metrics.get("map@20", float("nan")):.5f}`
- Recall@1: `{metrics.get("recall@1", float("nan")):.5f}`
- Recall@5: `{metrics.get("recall@5", float("nan")):.5f}`
- Recall@10: `{metrics.get("recall@10", float("nan")):.5f}`
- Recall@20: `{metrics.get("recall@20", float("nan")):.5f}`
- Embedding throughput: `{final["embedding_throughput_per_second"]:.2f}` listings/second
- Parameters: `{run["model"]["parameter_count"]:,}`
- Serialized checkpoint: `{run["model"]["checkpoint_bytes"]:,}` bytes

Phase 2 reference points on the same real validation split are pHash mAP@20 `0.2895` and ORB
mAP@20 `0.6638`. A smoke or bounded pilot run is not a fair claim against those full-data
baselines.

## Training curve

| Epoch | Train loss | Validation {run["selection"]["metric"]} | Seconds |
|---:|---:|---:|---:|
{history_rows}

## Reproducibility

- Seed: `{run["provenance"]["seed"]}`
- Git commit: `{run["provenance"]["git_commit"]}`
- Dirty worktree at run time: `{run["provenance"]["git_dirty"]}`
- Split manifest SHA-256: `{run["provenance"]["manifest_sha256"]}`
- Device: `{run["provenance"]["device"]}`
- Initialization: `{run["model"]["initialization_policy"]}`
- Normalization: `{run["model"]["normalization_policy"]}`
"""


def run_scratch_image_experiment(
    config_path: Path, *, progress_updates_per_epoch: int = 5
) -> dict[str, Any]:
    """Train from random initialization and select a checkpoint on validation retrieval."""
    if progress_updates_per_epoch < 0:
        raise ValueError("progress_updates_per_epoch must be non-negative")
    config = load_image_experiment_config(config_path)
    seed_everything(config.seed, deterministic=config.training.deterministic)
    device = _resolve_device(config.training.device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    splits = load_splits(config.data.metadata_csv, config.data.split_manifest)
    train_split = splits["train"]
    validation_split = splits[config.evaluation.tune_split]

    train_preprocessor = ImagePreprocessor(config.image_size, training=True, seed=config.seed)
    validation_preprocessor = ImagePreprocessor(config.image_size, training=False, seed=config.seed)
    train_dataset = ProductImageDataset.for_split(
        train_split, config.data.image_dir, train_preprocessor
    )
    validation_dataset = ProductImageDataset.for_split(
        validation_split, config.data.image_dir, validation_preprocessor
    )
    sampler = ProductBatchSampler(
        train_dataset.labels,
        products_per_batch=config.training.products_per_batch,
        samples_per_product=config.training.samples_per_product,
        batches_per_epoch=config.training.batches_per_epoch,
        seed=config.seed,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=sampler,
        num_workers=config.training.num_workers,
        pin_memory=device.type == "cuda",
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=sampler.batch_size,
        shuffle=False,
        num_workers=config.training.num_workers,
        pin_memory=device.type == "cuda",
    )
    LOGGER.info(
        "run started: device=%s epochs=%d batches/epoch=%d batch_size=%d image_size=%d",
        device,
        config.training.epochs,
        len(train_loader),
        sampler.batch_size,
        config.image_size,
    )
    LOGGER.info(
        "data ready: train=%d validation=%d; test evaluation disabled",
        len(train_dataset),
        len(validation_dataset),
    )

    model = ScratchResidualImageEncoder(config.model_spec).to(device)
    loss_function = SupervisedContrastiveLoss(config.training.temperature)
    optimizer = AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=config.training.epochs,
        eta_min=config.training.minimum_learning_rate,
    )
    checkpoint_path = config.artifacts.root / "best.pt"
    latest_path = config.artifacts.root / "latest.pt"
    history: list[dict[str, Any]] = []
    start_epoch, best_metric, best_epoch = 0, float("-inf"), -1
    if config.training.resume_from is not None:
        start_epoch, best_metric, best_epoch, history = _load_resume(
            config.training.resume_from, config, model, optimizer, scheduler
        )
        LOGGER.info("resumed training at epoch %d", start_epoch + 1)

    provenance = _provenance(config, device)
    run_started = time.perf_counter()
    epochs_without_improvement = 0
    progress_milestones = _progress_milestones(len(train_loader), progress_updates_per_epoch)
    for epoch in range(start_epoch, config.training.epochs):
        epoch_started = time.perf_counter()
        train_started = epoch_started
        LOGGER.info("epoch %d/%d: training", epoch + 1, config.training.epochs)
        sampler.set_epoch(epoch)
        train_dataset.set_epoch(epoch)
        model.train()
        total_loss = 0.0
        listings_seen: set[str] = set()
        groups_seen: set[int] = set()
        for batch_index, raw_batch in enumerate(train_loader):
            batch = cast(Mapping[str, Any], raw_batch)
            images = _tensor_batch(batch, "image", device)
            labels = _tensor_batch(batch, "label", device)
            optimizer.zero_grad(set_to_none=True)
            embeddings = model(images)
            loss = loss_function(embeddings, labels)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite training loss at epoch {epoch}")
            loss.backward()
            if epoch == start_epoch and batch_index == 0:
                _verify_finite_gradients(model)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.gradient_clip_norm)
            optimizer.step()
            total_loss += float(loss.detach().cpu())
            listings_seen.update(_posting_ids(batch))
            groups_seen.update(int(value) for value in labels.detach().cpu().tolist())
            completed_batches = batch_index + 1
            if completed_batches in progress_milestones:
                train_elapsed = time.perf_counter() - train_started
                eta_seconds = (
                    train_elapsed / completed_batches * (len(train_loader) - completed_batches)
                )
                LOGGER.info(
                    "epoch %d/%d: train %d/%d (%d%%) loss=%.5f elapsed=%s eta=%s",
                    epoch + 1,
                    config.training.epochs,
                    completed_batches,
                    len(train_loader),
                    round(100 * completed_batches / len(train_loader)),
                    total_loss / completed_batches,
                    _format_duration(train_elapsed),
                    _format_duration(eta_seconds),
                )
        scheduler.step()

        LOGGER.info("epoch %d/%d: validating", epoch + 1, config.training.epochs)
        validation_metrics, threshold, extraction_seconds, _, _, search_latency = (
            _validation_result(model, validation_loader, validation_split, config, device)
        )
        current_metric = validation_metrics[config.evaluation.checkpoint_metric]
        epoch_record: dict[str, Any] = {
            "epoch": epoch,
            "train_loss": total_loss / len(train_loader),
            "learning_rate": optimizer.param_groups[0]["lr"],
            "epoch_seconds": time.perf_counter() - epoch_started,
            "unique_train_listings": len(listings_seen),
            "unique_train_groups": len(groups_seen),
            "validation": validation_metrics,
            "validation_threshold": threshold,
            "embedding_extraction_seconds": extraction_seconds,
            "search_latency": search_latency,
        }
        history.append(epoch_record)
        improved = current_metric > best_metric
        if improved:
            best_metric, best_epoch = current_metric, epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        payload = _checkpoint_payload(
            config=config,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            best_metric=best_metric,
            best_epoch=best_epoch,
            history=history,
        )
        _save_checkpoint_atomic(latest_path, payload)
        if improved:
            _save_checkpoint_atomic(checkpoint_path, payload)
        _write_text_atomic(
            config.artifacts.root / "history.json",
            json.dumps(history, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        )
        LOGGER.info(
            "epoch %d/%d complete: loss=%.5f %s=%.5f best=%.5f checkpoint=%s elapsed=%s",
            epoch + 1,
            config.training.epochs,
            epoch_record["train_loss"],
            config.evaluation.checkpoint_metric,
            current_metric,
            best_metric,
            "best" if improved else "latest",
            _format_duration(epoch_record["epoch_seconds"]),
        )
        if epochs_without_improvement >= config.training.early_stopping_patience:
            LOGGER.info(
                "early stopping after %d epochs without validation improvement",
                epochs_without_improvement,
            )
            break

    LOGGER.info("loading best checkpoint from epoch %d for final validation", best_epoch + 1)
    selected = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(selected["model_state"])
    (
        validation_metrics,
        threshold,
        extraction_seconds,
        validation_embeddings,
        validation_ranking,
        search_latency,
    ) = _validation_result(model, validation_loader, validation_split, config, device)
    checkpoint_bytes = checkpoint_path.stat().st_size
    run: dict[str, Any] = {
        "pipeline_version": "phase3.scratch_image_training.v1",
        "provenance": provenance,
        "data": {name: len(split.items) for name, split in splits.items()},
        "model": {
            "parameter_count": model.parameter_count,
            "embedding_dim": config.model_spec.embedding_dim,
            "tensor_shapes": model.tensor_shapes(config.image_size),
            "checkpoint_bytes": checkpoint_bytes,
            "embedding_storage_bytes": validation_embeddings.nbytes,
            "initialization_policy": model.initialization_policy,
            "normalization_policy": ImagePreprocessor.normalization_policy,
        },
        "selection": {
            "split": "validation",
            "metric": config.evaluation.checkpoint_metric,
            "best_epoch": int(selected["best_epoch"]),
            "best_metric": float(selected["best_metric"]),
        },
        "history": cast(list[dict[str, Any]], selected["history"]),
        "validation": {
            "retrieval": validation_metrics,
            "stratified_retrieval": stratified_retrieval_metrics(
                validation_ranking,
                validation_split,
                config.evaluation.recall_at,
                config.evaluation.average_precision_at,
            ),
            "similarity_diagnostics": similarity_diagnostics(
                tuple(item.posting_id for item in validation_split.items),
                validation_embeddings,
                validation_split.label_by_id,
                seed=config.seed,
            ),
            "selected_pair_threshold": threshold,
            "embedding_extraction_seconds": extraction_seconds,
            "embedding_throughput_per_second": len(validation_dataset) / extraction_seconds,
            "search_latency": search_latency,
        },
        "test": {"status": "disabled_until_checkpoint_and_protocol_are_frozen"},
        "efficiency": {
            "wall_time_seconds": time.perf_counter() - run_started,
            "cuda_peak_allocated_bytes": (
                torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
            ),
        },
    }
    metrics_path = config.artifacts.root / "metrics.json"
    _write_text_atomic(
        metrics_path, json.dumps(run, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )
    _write_text_atomic(
        config.artifacts.root / "nearest_neighbor_review.json",
        json.dumps(
            nearest_neighbor_review(validation_ranking, validation_split),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
    )
    _write_text_atomic(config.artifacts.report, _render_report(run))
    LOGGER.info(
        "run complete: best_epoch=%d %s=%.5f wall_time=%s metrics=%s",
        run["selection"]["best_epoch"] + 1,
        config.evaluation.checkpoint_metric,
        run["selection"]["best_metric"],
        _format_duration(run["efficiency"]["wall_time_seconds"]),
        metrics_path,
    )
    return {
        "status": "complete",
        "checkpoint": str(checkpoint_path),
        "metrics": str(metrics_path),
        "report": str(config.artifacts.report),
        "best_epoch": run["selection"]["best_epoch"],
        "validation_metric": run["selection"]["best_metric"],
        "test_status": run["test"]["status"],
    }
