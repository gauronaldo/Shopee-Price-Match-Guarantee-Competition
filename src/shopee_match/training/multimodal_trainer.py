"""Configuration-driven Phase 5 training over cached frozen modality embeddings."""

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

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor, nn
from torch.nn import functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from shopee_match.errors import ConfigurationError, DataValidationError
from shopee_match.evaluation.embedding_retrieval import rank_cosine_embeddings_profiled
from shopee_match.evaluation.multimodal_retrieval import (
    rerank_with_pair_head,
    select_simple_score_fusion,
    unimodal_rankings,
)
from shopee_match.evaluation.protocol import (
    EvaluationSplit,
    load_splits,
    retrieval_metrics,
    select_threshold,
)
from shopee_match.losses import SupervisedContrastiveLoss
from shopee_match.models import LearnedMultimodalFusion, balanced_pair_indices
from shopee_match.reproducibility import seed_everything
from shopee_match.training.multimodal_config import (
    MultimodalExperimentConfig,
    load_multimodal_experiment_config,
)
from shopee_match.training.multimodal_data import load_cached_multimodal_split
from shopee_match.training.sampling import ProductBatchSampler

LOGGER = logging.getLogger(__name__)
FloatArray = NDArray[np.float32]
SOURCE_METRIC_REPRODUCTION_TOLERANCE = 1e-4


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


def _resolve_device(requested: str) -> torch.device:
    if requested == "cuda" and not torch.cuda.is_available():
        raise ConfigurationError("training.device=cuda but CUDA is not available")
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def _format_duration(seconds: float) -> str:
    rounded = max(0, round(seconds))
    minutes, secs = divmod(rounded, 60)
    return f"{minutes}m {secs:02d}s" if minutes else f"{secs}s"


def _progress_milestones(total_batches: int, updates: int) -> frozenset[int]:
    if total_batches <= 0 or updates < 0:
        raise ValueError("progress inputs are invalid")
    if updates == 0:
        return frozenset()
    return frozenset(
        min(total_batches, max(1, round(total_batches * step / updates)))
        for step in range(1, updates + 1)
    )


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


def _save_checkpoint_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _write_text_atomic(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def _verify_finite_gradients(model: nn.Module) -> None:
    non_finite = [
        name
        for name, parameter in model.named_parameters()
        if parameter.grad is not None and not torch.isfinite(parameter.grad).all()
    ]
    if non_finite:
        raise RuntimeError(f"Non-finite fusion gradients: {non_finite[:5]}")
    if not any(parameter.grad is not None for parameter in model.parameters()):
        raise RuntimeError("Fusion gradient gate found no gradients")


def extract_joint_embeddings(
    model: LearnedMultimodalFusion,
    loader: DataLoader[dict[str, Tensor | str]],
    device: torch.device,
) -> tuple[tuple[str, ...], FloatArray, FloatArray, FloatArray, float]:
    """Extract aligned base and learned embeddings in deterministic dataset order."""
    model.eval()
    posting_ids: list[str] = []
    image_parts: list[FloatArray] = []
    text_parts: list[FloatArray] = []
    joint_parts: list[FloatArray] = []
    started = time.perf_counter()
    with torch.inference_mode():
        for raw_batch in loader:
            batch = cast(Mapping[str, Any], raw_batch)
            image = _tensor_batch(batch, "image_embedding", device)
            text = _tensor_batch(batch, "text_embedding", device)
            image_parts.append(cast(FloatArray, image.cpu().numpy()))
            text_parts.append(cast(FloatArray, text.cpu().numpy()))
            joint_parts.append(cast(FloatArray, model(image, text).cpu().numpy()))
            posting_ids.extend(_posting_ids(batch))
    if not joint_parts:
        raise DataValidationError("Cannot evaluate an empty multimodal split")
    return (
        tuple(posting_ids),
        np.concatenate(image_parts),
        np.concatenate(text_parts),
        np.concatenate(joint_parts),
        time.perf_counter() - started,
    )


def _base_validation(
    posting_ids: tuple[str, ...],
    image: FloatArray,
    text: FloatArray,
    split: EvaluationSplit,
    config: MultimodalExperimentConfig,
) -> dict[str, Any]:
    image_ranking, text_ranking = unimodal_rankings(
        posting_ids, image, text, config.evaluation.candidate_k
    )
    image_metrics = retrieval_metrics(
        image_ranking,
        split.label_by_id,
        config.evaluation.recall_at,
        config.evaluation.average_precision_at,
    )
    text_metrics = retrieval_metrics(
        text_ranking,
        split.label_by_id,
        config.evaluation.recall_at,
        config.evaluation.average_precision_at,
    )
    selected_weight, simple_ranking, simple_metrics, trials = select_simple_score_fusion(
        posting_ids,
        image,
        text,
        split.label_by_id,
        image_weights=config.evaluation.simple_fusion_image_weights,
        candidate_k=config.evaluation.candidate_k,
        recall_at=config.evaluation.recall_at,
        average_precision_at=config.evaluation.average_precision_at,
    )
    metric = config.evaluation.checkpoint_metric
    expected_image = config.frozen.image_config.checkpoint.validation_metric_value
    expected_text = config.frozen.text_config.checkpoint.validation_metric_value
    image_delta = image_metrics[metric] - expected_image
    text_delta = text_metrics[metric] - expected_text
    if not np.isclose(
        image_metrics[metric],
        expected_image,
        rtol=0.0,
        atol=SOURCE_METRIC_REPRODUCTION_TOLERANCE,
    ):
        raise DataValidationError("Cached image validation metric differs from its frozen source")
    if not np.isclose(
        text_metrics[metric],
        expected_text,
        rtol=0.0,
        atol=SOURCE_METRIC_REPRODUCTION_TOLERANCE,
    ):
        raise DataValidationError("Cached text validation metric differs from its frozen source")
    return {
        "source_reproduction": {
            "absolute_tolerance": SOURCE_METRIC_REPRODUCTION_TOLERANCE,
            "image_metric_delta": image_delta,
            "text_metric_delta": text_delta,
        },
        "image_only": image_metrics,
        "text_only": text_metrics,
        "simple_score_fusion": {
            "selected_image_weight": selected_weight,
            "retrieval": simple_metrics,
            "selected_pair_threshold": select_threshold(simple_ranking, split.label_by_id),
            "weight_trials": trials,
        },
    }


def _learned_validation(
    model: LearnedMultimodalFusion,
    loader: DataLoader[dict[str, Tensor | str]],
    split: EvaluationSplit,
    config: MultimodalExperimentConfig,
    device: torch.device,
) -> tuple[dict[str, Any], FloatArray]:
    posting_ids, _image, _text, joint, extraction_seconds = extract_joint_embeddings(
        model, loader, device
    )
    cosine_ranking, search_latency = rank_cosine_embeddings_profiled(
        posting_ids, joint, config.evaluation.candidate_k
    )
    cosine_metrics = retrieval_metrics(
        cosine_ranking,
        split.label_by_id,
        config.evaluation.recall_at,
        config.evaluation.average_precision_at,
    )
    pair_ranking = rerank_with_pair_head(model, posting_ids, joint, cosine_ranking, device)
    pair_metrics = retrieval_metrics(
        pair_ranking,
        split.label_by_id,
        config.evaluation.recall_at,
        config.evaluation.average_precision_at,
    )
    return (
        {
            "learned_fusion": {
                "retrieval": cosine_metrics,
                "selected_pair_threshold": select_threshold(cosine_ranking, split.label_by_id),
            },
            "pair_head_rerank": {
                "candidate_source": "learned_fusion_top_k",
                "retrieval": pair_metrics,
                "selected_pair_threshold": select_threshold(pair_ranking, split.label_by_id),
            },
            "embedding_extraction_seconds": extraction_seconds,
            "embedding_throughput_per_second": len(posting_ids) / extraction_seconds,
            "search_latency": search_latency,
        },
        joint,
    )


def _checkpoint_payload(
    config: MultimodalExperimentConfig,
    model: LearnedMultimodalFusion,
    optimizer: AdamW,
    scheduler: CosineAnnealingLR,
    *,
    epoch: int,
    best_metric: float,
    best_epoch: int,
    history: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "checkpoint_version": "phase5.scratch_multimodal_checkpoint.v1",
        "epoch": epoch,
        "best_metric": best_metric,
        "best_epoch": best_epoch,
        "checkpoint_metric": config.evaluation.checkpoint_metric,
        "checkpoint_target": config.evaluation.checkpoint_target,
        "model_spec": {
            "image_embedding_dim": config.model_spec.image_embedding_dim,
            "text_embedding_dim": config.model_spec.text_embedding_dim,
            "fusion_hidden_dim": config.model_spec.fusion_hidden_dim,
            "joint_embedding_dim": config.model_spec.joint_embedding_dim,
            "pair_hidden_dim": config.model_spec.pair_hidden_dim,
            "dropout": config.model_spec.dropout,
            "fusion_mode": config.model_spec.fusion_mode,
            "base_image_weight": config.model_spec.base_image_weight,
            "residual_scale": config.model_spec.residual_scale,
        },
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),  # type: ignore[no-untyped-call]
        "history": history,
        "seed": config.seed,
        "source_checkpoints": {
            "image_sha256": config.frozen.image_config.checkpoint.sha256,
            "text_sha256": config.frozen.text_config.checkpoint.sha256,
        },
        "encoders_frozen": True,
    }


def _render_report(run: dict[str, Any]) -> str:
    validation = run["validation"]
    base = validation["base_ablations"]
    learned = validation["selected_checkpoint"]
    history_rows = "\n".join(
        f"| {row['epoch'] + 1} | {row['train_loss']:.5f} | "
        f"{row['contrastive_loss']:.5f} | {row['pair_bce_loss']:.5f} | "
        f"{row['validation_map']:.5f} |"
        for row in run["history"]
    )
    metric = run["selection"]["metric"]
    target = run["selection"]["target"]
    simple = base["simple_score_fusion"]["retrieval"]
    joint = learned["learned_fusion"]["retrieval"]
    pair = learned["pair_head_rerank"]["retrieval"]
    stage = run["experiment_stage"]
    best_epoch = run["selection"]["best_epoch"]
    selected_label = "initialization" if best_epoch < 0 else f"epoch {best_epoch + 1}"
    if stage == "training":
        summary = run["training_summary"]
        completed = summary["completed_epochs"]
        configured = summary["configured_epochs"]
        interpretation = f"""This full validation-only run completed `{completed}` of
`{configured}` configured epochs and stopped early after the selected metric failed to improve.
The best checkpoint is retained independently of the lower later training loss. Held-out test
remains disabled until the checkpoint, threshold, and protocol are frozen."""
    else:
        interpretation = f"""This `{stage}` run is an engineering gate, not the final Phase 5
benchmark. It verifies frozen-source reproducibility, loss/gradient flow, checkpoint selection,
modality ablations, pair-head behavior, and validation-only evaluation before a full fusion run is
approved."""
    return f"""# Scratch multimodal fusion implementation

## Experiment contract

The Phase 5 `{stage}` experiment reuses the frozen scratch image and text encoders from Phases 3-4.
Their deterministic embeddings are cached once. Only the randomly initialized fusion projection
and symmetric pair head are optimized; held-out test remains disabled.

## Validation comparison

| Method | mAP@20 | Recall@20 |
|---|---:|---:|
| Frozen image only | {base["image_only"]["map@20"]:.5f} | {base["image_only"]["recall@20"]:.5f} |
| Frozen text only | {base["text_only"]["map@20"]:.5f} | {base["text_only"]["recall@20"]:.5f} |
| Simple score fusion | {simple["map@20"]:.5f} | {simple["recall@20"]:.5f} |
| Learned fusion | {joint["map@20"]:.5f} | {joint["recall@20"]:.5f} |
| Pair-head rerank | {pair["map@20"]:.5f} | {pair["recall@20"]:.5f} |

- Selected simple-fusion image weight: `{base["simple_score_fusion"]["selected_image_weight"]:.2f}`
- Selected checkpoint: `{selected_label}` by validation `{target}.{metric}`
- Trainable parameters: `{run["model"]["parameter_count"]:,}`
- Test status: `disabled_until_checkpoint_and_protocol_are_frozen`

## Training history

| Epoch | Total loss | Contrastive | Pair BCE | Validation mAP@20 |
|---:|---:|---:|---:|---:|
{history_rows}

## Interpretation

{interpretation}
"""


def run_multimodal_experiment(
    config_path: Path, *, progress_updates_per_epoch: int = 4
) -> dict[str, object]:
    """Train the Phase 5 fusion/pair heads without updating either source encoder."""
    if progress_updates_per_epoch < 0:
        raise ValueError("progress_updates_per_epoch must be non-negative")
    config = load_multimodal_experiment_config(config_path)
    seed_everything(config.seed, deterministic=config.training.deterministic)
    device = _resolve_device(config.training.device)
    splits = load_splits(config.data.metadata_csv, config.data.split_manifest)
    train_dataset = load_cached_multimodal_split(config, "train")
    validation_dataset = load_cached_multimodal_split(config, "validation")
    sampler = ProductBatchSampler(
        train_dataset.labels,
        products_per_batch=config.training.products_per_batch,
        samples_per_product=config.training.samples_per_product,
        batches_per_epoch=config.training.batches_per_epoch,
        seed=config.seed,
    )
    train_loader = DataLoader(train_dataset, batch_sampler=sampler, num_workers=0)
    validation_loader = DataLoader(validation_dataset, batch_size=512, shuffle=False, num_workers=0)
    model = LearnedMultimodalFusion(config.model_spec).to(device)
    contrastive_loss = SupervisedContrastiveLoss(config.loss.temperature)
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

    LOGGER.info(
        "Phase 5 run: device=%s train=%d validation=%d epochs=%d batches/epoch=%d test=disabled",
        device,
        len(train_dataset),
        len(validation_dataset),
        config.training.epochs,
        len(train_loader),
    )
    model.eval()
    posting_ids, image, text, _joint, _seconds = extract_joint_embeddings(
        model, validation_loader, device
    )
    base_validation = _base_validation(posting_ids, image, text, splits["validation"], config)
    LOGGER.info(
        "base ablations: image=%.5f text=%.5f simple=%.5f weight=%.2f",
        base_validation["image_only"]["map@20"],
        base_validation["text_only"]["map@20"],
        base_validation["simple_score_fusion"]["retrieval"]["map@20"],
        base_validation["simple_score_fusion"]["selected_image_weight"],
    )

    best_path = config.artifacts.root / "best.pt"
    latest_path = config.artifacts.root / "latest.pt"
    history: list[dict[str, Any]] = []
    initial_validation, _initial_embeddings = _learned_validation(
        model, validation_loader, splits["validation"], config, device
    )
    best_metric = initial_validation[config.evaluation.checkpoint_target]["retrieval"][
        config.evaluation.checkpoint_metric
    ]
    best_epoch = -1
    _save_checkpoint_atomic(
        best_path,
        _checkpoint_payload(
            config,
            model,
            optimizer,
            scheduler,
            epoch=-1,
            best_metric=best_metric,
            best_epoch=best_epoch,
            history=history,
        ),
    )
    LOGGER.info("%s initialization: map@20=%.5f", config.evaluation.checkpoint_target, best_metric)
    epochs_without_improvement = 0
    run_started = time.perf_counter()
    milestones = _progress_milestones(len(train_loader), progress_updates_per_epoch)
    for epoch in range(config.training.epochs):
        epoch_started = time.perf_counter()
        sampler.set_epoch(epoch)
        model.train()
        total_loss = total_contrastive = total_pair = 0.0
        LOGGER.info("epoch %d/%d: training fusion and pair head", epoch + 1, config.training.epochs)
        for batch_index, raw_batch in enumerate(train_loader):
            batch = cast(Mapping[str, Any], raw_batch)
            image_batch = _tensor_batch(batch, "image_embedding", device)
            text_batch = _tensor_batch(batch, "text_embedding", device)
            labels = _tensor_batch(batch, "label", device)
            optimizer.zero_grad(set_to_none=True)
            joint = model(image_batch, text_batch)
            supcon = contrastive_loss(joint, labels)
            pair_generator = torch.Generator(device="cpu")
            pair_generator.manual_seed(config.seed + epoch * len(train_loader) + batch_index)
            left, right, targets = balanced_pair_indices(
                labels,
                maximum_negative_ratio=config.loss.maximum_negative_ratio,
                generator=pair_generator,
            )
            pair_bce = F.binary_cross_entropy_with_logits(
                model.pair_logits(joint[left], joint[right]), targets
            )
            loss = (
                config.loss.supervised_contrastive_weight * supcon
                + config.loss.pair_bce_weight * pair_bce
            )
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite Phase 5 loss at epoch {epoch}")
            loss.backward()
            if epoch == 0 and batch_index == 0:
                _verify_finite_gradients(model)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.gradient_clip_norm)
            optimizer.step()
            total_loss += float(loss.detach().cpu())
            total_contrastive += float(supcon.detach().cpu())
            total_pair += float(pair_bce.detach().cpu())
            if batch_index + 1 in milestones:
                LOGGER.info(
                    "epoch %d/%d: batch %d/%d loss=%.5f",
                    epoch + 1,
                    config.training.epochs,
                    batch_index + 1,
                    len(train_loader),
                    total_loss / (batch_index + 1),
                )
        scheduler.step()
        LOGGER.info("epoch %d/%d: validation", epoch + 1, config.training.epochs)
        validation, _embeddings = _learned_validation(
            model, validation_loader, splits["validation"], config, device
        )
        current_metric = validation[config.evaluation.checkpoint_target]["retrieval"][
            config.evaluation.checkpoint_metric
        ]
        record = {
            "epoch": epoch,
            "train_loss": total_loss / len(train_loader),
            "contrastive_loss": total_contrastive / len(train_loader),
            "pair_bce_loss": total_pair / len(train_loader),
            "learning_rate": optimizer.param_groups[0]["lr"],
            "validation_map": current_metric,
            "validation": validation,
            "epoch_seconds": time.perf_counter() - epoch_started,
        }
        history.append(record)
        improved = current_metric > best_metric
        if improved:
            best_metric, best_epoch, epochs_without_improvement = current_metric, epoch, 0
        else:
            epochs_without_improvement += 1
        payload = _checkpoint_payload(
            config,
            model,
            optimizer,
            scheduler,
            epoch=epoch,
            best_metric=best_metric,
            best_epoch=best_epoch,
            history=history,
        )
        _save_checkpoint_atomic(latest_path, payload)
        if improved:
            _save_checkpoint_atomic(best_path, payload)
        LOGGER.info(
            "epoch %d/%d complete: loss=%.5f map@20=%.5f best=%.5f checkpoint=%s elapsed=%s",
            epoch + 1,
            config.training.epochs,
            record["train_loss"],
            current_metric,
            best_metric,
            "best" if improved else "latest",
            _format_duration(record["epoch_seconds"]),
        )
        if epochs_without_improvement >= config.training.early_stopping_patience:
            LOGGER.info("early stopping after %d unimproved epochs", epochs_without_improvement)
            break

    selected = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(selected["model_state"])
    selected_validation, joint_embeddings = _learned_validation(
        model, validation_loader, splits["validation"], config, device
    )
    commit, dirty = _git_state()
    run: dict[str, Any] = {
        "pipeline_version": "phase5.scratch_multimodal_training.v1",
        "experiment_stage": config.artifacts.root.name,
        "provenance": {
            "config_sha256": _sha256(config.config_path),
            "model_config_sha256": _sha256(config.model_config_path),
            "manifest_sha256": _sha256(config.data.split_manifest),
            "git_commit": commit,
            "git_dirty": dirty,
            "seed": config.seed,
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "device": str(device),
        },
        "frozen_sources": {
            "image_checkpoint_sha256": config.frozen.image_config.checkpoint.sha256,
            "text_checkpoint_sha256": config.frozen.text_config.checkpoint.sha256,
            "image_encoder_updated": False,
            "text_encoder_updated": False,
        },
        "data": {name: len(split.items) for name, split in splits.items()},
        "model": {
            "parameter_count": model.parameter_count,
            "joint_embedding_dim": config.model_spec.joint_embedding_dim,
            "checkpoint_bytes": best_path.stat().st_size,
            "joint_embedding_storage_bytes": joint_embeddings.nbytes,
            "initialization_policy": model.initialization_policy,
        },
        "loss": {
            "supervised_contrastive_weight": config.loss.supervised_contrastive_weight,
            "pair_bce_weight": config.loss.pair_bce_weight,
            "temperature": config.loss.temperature,
            "maximum_negative_ratio": config.loss.maximum_negative_ratio,
        },
        "selection": {
            "split": "validation",
            "metric": config.evaluation.checkpoint_metric,
            "target": config.evaluation.checkpoint_target,
            "best_epoch": int(selected["best_epoch"]),
            "best_metric": float(selected["best_metric"]),
        },
        "training_summary": {
            "configured_epochs": config.training.epochs,
            "completed_epochs": len(history),
            "last_epoch": int(history[-1]["epoch"]),
            "best_epoch": int(selected["best_epoch"]),
            "stopped_early": len(history) < config.training.epochs,
        },
        "history": history,
        "validation": {
            "base_ablations": base_validation,
            "initialization": initial_validation,
            "selected_checkpoint": selected_validation,
        },
        "test": {"status": "disabled_until_checkpoint_and_protocol_are_frozen"},
        "efficiency": {"wall_time_seconds": time.perf_counter() - run_started},
    }
    metrics_path = config.artifacts.root / "metrics.json"
    _write_text_atomic(metrics_path, json.dumps(run, indent=2, sort_keys=True) + "\n")
    _write_text_atomic(config.artifacts.report, _render_report(run))
    return {
        "status": "complete",
        "checkpoint": str(best_path),
        "metrics": str(metrics_path),
        "report": str(config.artifacts.report),
        "best_epoch": run["selection"]["best_epoch"],
        "validation_metric": run["selection"]["best_metric"],
        "test_status": run["test"]["status"],
    }


def refresh_multimodal_training_report(config_path: Path) -> dict[str, object]:
    """Refresh metrics/report metadata from checkpoints without training or evaluation."""
    config = load_multimodal_experiment_config(config_path)
    best_path = config.artifacts.root / "best.pt"
    latest_path = config.artifacts.root / "latest.pt"
    metrics_path = config.artifacts.root / "metrics.json"
    best = torch.load(best_path, map_location="cpu", weights_only=False)
    latest = torch.load(latest_path, map_location="cpu", weights_only=False)
    run = cast(dict[str, Any], json.loads(metrics_path.read_text(encoding="utf-8")))
    if (
        best.get("checkpoint_version") != "phase5.scratch_multimodal_checkpoint.v1"
        or latest.get("checkpoint_version") != "phase5.scratch_multimodal_checkpoint.v1"
    ):
        raise ConfigurationError("Unsupported multimodal checkpoint version")
    if best.get("best_epoch") != latest.get("best_epoch") or not np.isclose(
        float(best.get("best_metric", float("nan"))),
        float(latest.get("best_metric", float("nan"))),
        rtol=0.0,
        atol=1e-12,
    ):
        raise ConfigurationError("Best and latest multimodal checkpoint selection differs")
    history = cast(list[dict[str, Any]], latest.get("history", []))
    if not history or int(history[-1]["epoch"]) != int(latest["epoch"]):
        raise ConfigurationError("Latest multimodal checkpoint history is incomplete")
    run["history"] = history
    run["training_summary"] = {
        "configured_epochs": config.training.epochs,
        "completed_epochs": len(history),
        "last_epoch": int(latest["epoch"]),
        "best_epoch": int(best["best_epoch"]),
        "stopped_early": len(history) < config.training.epochs,
    }
    _write_text_atomic(metrics_path, json.dumps(run, indent=2, sort_keys=True) + "\n")
    _write_text_atomic(config.artifacts.report, _render_report(run))
    return {
        "status": "complete",
        "action": "report_refreshed_without_training_or_evaluation",
        "metrics": str(metrics_path),
        "report": str(config.artifacts.report),
        "completed_epochs": len(history),
        "best_epoch": int(best["best_epoch"]),
    }
