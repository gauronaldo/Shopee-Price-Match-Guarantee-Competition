"""Configuration-driven Phase 4 scratch TextCNN training and validation retrieval."""

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
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from shopee_match.errors import ConfigurationError, DataValidationError
from shopee_match.evaluation.embedding_retrieval import (
    rank_cosine_embeddings_profiled,
    similarity_diagnostics,
)
from shopee_match.evaluation.protocol import (
    EvaluationSplit,
    Ranking,
    load_splits,
    retrieval_metrics,
    select_threshold,
)
from shopee_match.evaluation.text_retrieval import (
    stratified_text_retrieval_metrics,
    title_length_summary,
)
from shopee_match.features.text import normalize_title
from shopee_match.losses import SupervisedContrastiveLoss
from shopee_match.models import ScratchTextCNN
from shopee_match.reproducibility import seed_everything
from shopee_match.training.sampling import ProductBatchSampler
from shopee_match.training.text_config import TextExperimentConfig, load_text_experiment_config
from shopee_match.training.text_data import CharacterVocabulary, ProductTextDataset

LOGGER = logging.getLogger(__name__)
FloatArray = NDArray[np.floating[Any]]


def _format_duration(seconds: float) -> str:
    rounded = max(0, round(seconds))
    hours, remainder = divmod(rounded, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes:02d}m"
    if minutes:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"


def _progress_milestones(total_batches: int, updates: int) -> frozenset[int]:
    if total_batches <= 0 or updates < 0:
        raise ValueError("total_batches must be positive and updates non-negative")
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
    missing = [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and parameter.grad is None
    ]
    non_finite = [
        name
        for name, parameter in model.named_parameters()
        if parameter.grad is not None and not torch.isfinite(parameter.grad).all()
    ]
    if missing or non_finite:
        raise RuntimeError(
            f"Gradient gate failed; missing={missing[:5]}, non_finite={non_finite[:5]}"
        )


def extract_text_embeddings(
    model: ScratchTextCNN,
    loader: DataLoader[dict[str, Tensor | str]],
    device: torch.device,
) -> tuple[tuple[str, ...], FloatArray, float]:
    """Extract normalized title embeddings in deterministic split order."""
    model.eval()
    posting_ids: list[str] = []
    parts: list[FloatArray] = []
    started = time.perf_counter()
    with torch.inference_mode():
        for raw_batch in loader:
            batch = cast(Mapping[str, Any], raw_batch)
            token_ids = _tensor_batch(batch, "token_ids", device)
            lengths = _tensor_batch(batch, "length", device)
            parts.append(model(token_ids, lengths).cpu().numpy())
            posting_ids.extend(_posting_ids(batch))
    elapsed = time.perf_counter() - started
    if not parts:
        raise DataValidationError("Cannot extract text embeddings from an empty split")
    return tuple(posting_ids), np.concatenate(parts, axis=0), elapsed


def _validation_result(
    model: ScratchTextCNN,
    loader: DataLoader[dict[str, Tensor | str]],
    split: EvaluationSplit,
    config: TextExperimentConfig,
    device: torch.device,
) -> tuple[dict[str, float], dict[str, float], float, FloatArray, Ranking, dict[str, float]]:
    LOGGER.info("validation: extracting text embeddings")
    posting_ids, embeddings, extraction_seconds = extract_text_embeddings(model, loader, device)
    LOGGER.info("validation: ranking %d listings and computing retrieval metrics", len(posting_ids))
    ranking, latency = rank_cosine_embeddings_profiled(
        posting_ids, embeddings, config.evaluation.candidate_k
    )
    metrics = retrieval_metrics(
        ranking,
        split.label_by_id,
        config.evaluation.recall_at,
        config.evaluation.average_precision_at,
    )
    return (
        metrics,
        select_threshold(ranking, split.label_by_id),
        extraction_seconds,
        embeddings,
        ranking,
        latency,
    )


def _unknown_character_rate(split: EvaluationSplit, vocabulary: CharacterVocabulary) -> float:
    known = set(vocabulary.tokens[2:])
    characters = [character for item in split.items for character in normalize_title(item.title)]
    return (
        sum(character not in known for character in characters) / len(characters)
        if characters
        else 0.0
    )


def _checkpoint_payload(
    config: TextExperimentConfig,
    model: ScratchTextCNN,
    vocabulary: CharacterVocabulary,
    optimizer: AdamW,
    scheduler: CosineAnnealingLR,
    epoch: int,
    best_metric: float,
    best_epoch: int,
    history: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "checkpoint_version": "phase4.scratch_text_checkpoint.v1",
        "epoch": epoch,
        "best_metric": best_metric,
        "best_epoch": best_epoch,
        "checkpoint_metric": config.evaluation.checkpoint_metric,
        "model_spec": {
            "character_embedding_dim": config.model_spec.character_embedding_dim,
            "convolution_channels": config.model_spec.convolution_channels,
            "kernel_sizes": config.model_spec.kernel_sizes,
            "projection_hidden_dim": config.model_spec.projection_hidden_dim,
            "embedding_dim": config.model_spec.embedding_dim,
            "dropout": config.model_spec.dropout,
        },
        "vocabulary": vocabulary.to_dict(),
        "maximum_length": config.tokenization.maximum_length,
        "seed": config.seed,
        "split_manifest_sha256": _sha256(config.data.split_manifest),
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),  # type: ignore[no-untyped-call]
        "history": history,
    }


def _render_report(run: dict[str, Any]) -> str:
    metrics = run["validation"]["retrieval"]
    rows = "\n".join(
        f"| {row['epoch']} | {row['train_loss']:.5f} | "
        f"{row['validation'][run['selection']['metric']]:.5f} | {row['epoch_seconds']:.2f} |"
        for row in run["history"]
    )
    return f"""# Scratch text encoder benchmark

## Experiment status

This Phase 4 character TextCNN was initialized randomly. Its vocabulary was fitted from training
titles only; no pretrained tokenizer, word embedding, language model, image feature, or test label
entered training or checkpoint selection.

## Validation result

- Selected epoch: `{run["selection"]["best_epoch"] + 1}` of `{len(run["history"])}`
- mAP@20: `{metrics.get("map@20", float("nan")):.5f}`
- Recall@1: `{metrics.get("recall@1", float("nan")):.5f}`
- Recall@5: `{metrics.get("recall@5", float("nan")):.5f}`
- Recall@10: `{metrics.get("recall@10", float("nan")):.5f}`
- Recall@20: `{metrics.get("recall@20", float("nan")):.5f}`
- Vocabulary size: `{run["vocabulary"]["size"]}`
- Validation unknown-character rate: `{run["vocabulary"]["validation_unknown_character_rate"]:.6f}`
- Parameters: `{run["model"]["parameter_count"]:,}`

The Phase 2 character TF-IDF validation reference is mAP@20 `0.8635`. Smoke and bounded pilot
runs are engineering evidence, not final claims against that full baseline.

## Training curve

| Epoch | Train loss | Validation {run["selection"]["metric"]} | Seconds |
|---:|---:|---:|---:|
{rows}
"""


def run_scratch_text_experiment(
    config_path: Path, *, progress_updates_per_epoch: int = 5
) -> dict[str, Any]:
    """Train a random-init TextCNN and select its checkpoint on validation retrieval."""
    if progress_updates_per_epoch < 0:
        raise ValueError("progress_updates_per_epoch must be non-negative")
    config = load_text_experiment_config(config_path)
    seed_everything(config.seed, deterministic=config.training.deterministic)
    device = _resolve_device(config.training.device)
    splits = load_splits(config.data.metadata_csv, config.data.split_manifest)
    train_split, validation_split = splits["train"], splits["validation"]
    vocabulary = CharacterVocabulary.fit(
        tuple(item.title for item in train_split.items),
        minimum_frequency=config.tokenization.minimum_frequency,
        maximum_size=config.tokenization.maximum_vocabulary_size,
    )
    train_dataset = ProductTextDataset(train_split, vocabulary, config.tokenization.maximum_length)
    validation_dataset = ProductTextDataset(
        validation_split, vocabulary, config.tokenization.maximum_length
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
    model = ScratchTextCNN(
        len(vocabulary.tokens), config.model_spec, padding_index=vocabulary.padding_index
    ).to(device)
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
    LOGGER.info(
        "run started: device=%s epochs=%d batches/epoch=%d batch_size=%d vocab=%d max_length=%d",
        device,
        config.training.epochs,
        len(train_loader),
        sampler.batch_size,
        len(vocabulary.tokens),
        config.tokenization.maximum_length,
    )
    LOGGER.info(
        "data ready: train=%d validation=%d; vocabulary fitted on train only; test disabled",
        len(train_dataset),
        len(validation_dataset),
    )

    checkpoint_path = config.artifacts.root / "best.pt"
    latest_path = config.artifacts.root / "latest.pt"
    history: list[dict[str, Any]] = []
    best_metric, best_epoch = float("-inf"), -1
    epochs_without_improvement = 0
    run_started = time.perf_counter()
    milestones = _progress_milestones(len(train_loader), progress_updates_per_epoch)
    for epoch in range(config.training.epochs):
        epoch_started = time.perf_counter()
        train_started = epoch_started
        sampler.set_epoch(epoch)
        model.train()
        total_loss = 0.0
        LOGGER.info("epoch %d/%d: training", epoch + 1, config.training.epochs)
        for batch_index, raw_batch in enumerate(train_loader):
            batch = cast(Mapping[str, Any], raw_batch)
            token_ids = _tensor_batch(batch, "token_ids", device)
            lengths = _tensor_batch(batch, "length", device)
            labels = _tensor_batch(batch, "label", device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_function(model(token_ids, lengths), labels)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite training loss at epoch {epoch}")
            loss.backward()
            if epoch == 0 and batch_index == 0:
                _verify_finite_gradients(model)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.gradient_clip_norm)
            optimizer.step()
            total_loss += float(loss.detach().cpu())
            completed = batch_index + 1
            if completed in milestones:
                elapsed = time.perf_counter() - train_started
                eta = elapsed / completed * (len(train_loader) - completed)
                LOGGER.info(
                    "epoch %d/%d: train %d/%d (%d%%) loss=%.5f elapsed=%s eta=%s",
                    epoch + 1,
                    config.training.epochs,
                    completed,
                    len(train_loader),
                    round(100 * completed / len(train_loader)),
                    total_loss / completed,
                    _format_duration(elapsed),
                    _format_duration(eta),
                )
        scheduler.step()
        LOGGER.info("epoch %d/%d: validating", epoch + 1, config.training.epochs)
        validation_metrics, threshold, extraction_seconds, _, _, latency = _validation_result(
            model, validation_loader, validation_split, config, device
        )
        current_metric = validation_metrics[config.evaluation.checkpoint_metric]
        record = {
            "epoch": epoch,
            "train_loss": total_loss / len(train_loader),
            "learning_rate": optimizer.param_groups[0]["lr"],
            "epoch_seconds": time.perf_counter() - epoch_started,
            "validation": validation_metrics,
            "validation_threshold": threshold,
            "embedding_extraction_seconds": extraction_seconds,
            "search_latency": latency,
        }
        history.append(record)
        improved = current_metric > best_metric
        if improved:
            best_metric, best_epoch, epochs_without_improvement = current_metric, epoch, 0
        else:
            epochs_without_improvement += 1
        payload = _checkpoint_payload(
            config, model, vocabulary, optimizer, scheduler, epoch, best_metric, best_epoch, history
        )
        _save_checkpoint_atomic(latest_path, payload)
        if improved:
            _save_checkpoint_atomic(checkpoint_path, payload)
        LOGGER.info(
            "epoch %d/%d complete: loss=%.5f %s=%.5f best=%.5f checkpoint=%s elapsed=%s",
            epoch + 1,
            config.training.epochs,
            record["train_loss"],
            config.evaluation.checkpoint_metric,
            current_metric,
            best_metric,
            "best" if improved else "latest",
            _format_duration(record["epoch_seconds"]),
        )
        if epochs_without_improvement >= config.training.early_stopping_patience:
            LOGGER.info("early stopping after %d unimproved epochs", epochs_without_improvement)
            break

    selected = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(selected["model_state"])
    metrics, threshold, extraction_seconds, embeddings, ranking, latency = _validation_result(
        model, validation_loader, validation_split, config, device
    )
    commit, dirty = _git_state()
    run: dict[str, Any] = {
        "pipeline_version": "phase4.scratch_text_training.v1",
        "provenance": {
            "config_sha256": _sha256(config.config_path),
            "model_config_sha256": _sha256(config.model_config_path),
            "manifest_sha256": _sha256(config.data.split_manifest),
            "metadata_sha256": _sha256(config.data.metadata_csv),
            "git_commit": commit,
            "git_dirty": dirty,
            "seed": config.seed,
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "device": str(device),
        },
        "data": {name: len(split.items) for name, split in splits.items()},
        "vocabulary": {
            "size": len(vocabulary.tokens),
            "source_split": "train",
            "validation_unknown_character_rate": _unknown_character_rate(
                validation_split, vocabulary
            ),
            "train_title_lengths": title_length_summary(
                train_split, config.tokenization.maximum_length
            ),
            "validation_title_lengths": title_length_summary(
                validation_split, config.tokenization.maximum_length
            ),
        },
        "model": {
            "parameter_count": model.parameter_count,
            "embedding_dim": config.model_spec.embedding_dim,
            "checkpoint_bytes": checkpoint_path.stat().st_size,
            "embedding_storage_bytes": embeddings.nbytes,
            "initialization_policy": model.initialization_policy,
        },
        "selection": {
            "split": "validation",
            "metric": config.evaluation.checkpoint_metric,
            "best_epoch": int(selected["best_epoch"]),
            "best_metric": float(selected["best_metric"]),
        },
        "history": history,
        "validation": {
            "retrieval": metrics,
            "stratified_retrieval": stratified_text_retrieval_metrics(
                ranking,
                validation_split,
                config.evaluation.recall_at,
                config.evaluation.average_precision_at,
            ),
            "similarity_diagnostics": similarity_diagnostics(
                tuple(item.posting_id for item in validation_split.items),
                embeddings,
                validation_split.label_by_id,
                seed=config.seed,
            ),
            "selected_pair_threshold": threshold,
            "embedding_extraction_seconds": extraction_seconds,
            "embedding_throughput_per_second": len(validation_dataset) / extraction_seconds,
            "search_latency": latency,
        },
        "test": {"status": "disabled_until_checkpoint_and_protocol_are_frozen"},
        "efficiency": {"wall_time_seconds": time.perf_counter() - run_started},
    }
    metrics_path = config.artifacts.root / "metrics.json"
    _write_text_atomic(
        config.artifacts.root / "vocabulary.json",
        json.dumps(vocabulary.to_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    _write_text_atomic(
        metrics_path, json.dumps(run, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
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
        "vocabulary": str(config.artifacts.root / "vocabulary.json"),
        "best_epoch": run["selection"]["best_epoch"],
        "validation_metric": run["selection"]["best_metric"],
        "test_status": run["test"]["status"],
    }


def refresh_text_training_report(config_path: Path) -> dict[str, Any]:
    """Restore complete history from latest.pt without training or evaluation."""
    config = load_text_experiment_config(config_path)
    checkpoint_path = config.artifacts.root / "best.pt"
    latest_path = config.artifacts.root / "latest.pt"
    metrics_path = config.artifacts.root / "metrics.json"
    best = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    latest = torch.load(latest_path, map_location="cpu", weights_only=False)
    run = cast(dict[str, Any], json.loads(metrics_path.read_text(encoding="utf-8")))
    if (
        best.get("checkpoint_version") != "phase4.scratch_text_checkpoint.v1"
        or latest.get("checkpoint_version") != "phase4.scratch_text_checkpoint.v1"
    ):
        raise ConfigurationError("Unsupported text checkpoint version")
    if best.get("best_epoch") != latest.get("best_epoch") or not np.isclose(
        float(best.get("best_metric", float("nan"))),
        float(latest.get("best_metric", float("nan"))),
        rtol=0.0,
        atol=1e-12,
    ):
        raise ConfigurationError("Best and latest checkpoint selection metadata differ")
    full_history = cast(list[dict[str, Any]], latest.get("history", []))
    if not full_history or int(full_history[-1]["epoch"]) != int(latest["epoch"]):
        raise ConfigurationError("Latest checkpoint does not contain a complete training history")
    selection = run.get("selection", {})
    if int(selection.get("best_epoch", -1)) != int(best["best_epoch"]) or not np.isclose(
        float(selection.get("best_metric", float("nan"))),
        float(best["best_metric"]),
        rtol=0.0,
        atol=1e-12,
    ):
        raise ConfigurationError("Training metrics selection differs from the best checkpoint")
    run["history"] = full_history
    run["training_summary"] = {
        "configured_epochs": config.training.epochs,
        "completed_epochs": len(full_history),
        "last_epoch": int(latest["epoch"]),
        "best_epoch": int(best["best_epoch"]),
        "stopped_early": len(full_history) < config.training.epochs,
    }
    _write_text_atomic(
        metrics_path, json.dumps(run, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )
    _write_text_atomic(config.artifacts.report, _render_report(run))
    return {
        "status": "complete",
        "action": "report_refreshed_without_training_or_evaluation",
        "metrics": str(metrics_path),
        "report": str(config.artifacts.report),
        "completed_epochs": len(full_history),
        "best_epoch": int(best["best_epoch"]),
    }
