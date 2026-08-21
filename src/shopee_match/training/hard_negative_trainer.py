"""Phase 6 fine-tuning with mixed random and mined hard negatives."""

from __future__ import annotations

import json
import logging
import os
import platform
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from shopee_match.errors import ConfigurationError, DataValidationError, OutputConflictError
from shopee_match.evaluation.embedding_retrieval import rank_cosine_embeddings_profiled
from shopee_match.evaluation.multimodal_retrieval import rerank_with_pair_head
from shopee_match.evaluation.protocol import (
    EvaluationSplit,
    Ranking,
    load_splits,
    precision_at_minimum_recall,
    retrieval_metrics,
    select_threshold,
)
from shopee_match.hashing import canonical_text_sha256, sha256_file
from shopee_match.losses import SupervisedContrastiveLoss
from shopee_match.models import LearnedMultimodalFusion, balanced_pair_indices
from shopee_match.reproducibility import seed_everything
from shopee_match.training.hard_negative_config import (
    HardNegativeExperimentConfig,
    load_hard_negative_experiment_config,
)
from shopee_match.training.hard_negative_data import (
    HardNegativeBatchProvider,
    has_variant_conflict,
    load_hard_negative_manifest,
)
from shopee_match.training.hard_negative_miner import load_phase5_source_model
from shopee_match.training.multimodal_data import (
    CachedMultimodalDataset,
    load_cached_multimodal_split,
)
from shopee_match.training.multimodal_trainer import (
    _format_duration,
    _git_state,
    _progress_milestones,
    _resolve_device,
    _save_checkpoint_atomic,
    _tensor_batch,
    _verify_finite_gradients,
    extract_joint_embeddings,
)
from shopee_match.training.sampling import ProductBatchSampler

LOGGER = logging.getLogger(__name__)


def _write_text_atomic(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def _load_mining_metadata(config: HardNegativeExperimentConfig) -> dict[str, Any]:
    try:
        metadata = cast(
            dict[str, Any],
            json.loads(config.artifacts.manifest_metadata.read_text(encoding="utf-8")),
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise DataValidationError("Cannot read mined-pair provenance metadata") from exc
    expected_source = {
        "config_sha256": config.source.multimodal_config_sha256,
        "checkpoint_sha256": config.source.checkpoint_sha256,
        "metrics_sha256": config.source.metrics_sha256,
        "split_manifest_sha256": sha256_file(
            config.source.experiment.data.split_manifest
        ),
    }
    if (
        metadata.get("pipeline_version") != "phase6.hard_negative_mining.v1"
        or metadata.get("split") != "train"
        or metadata.get("test_accessed") is not False
        or metadata.get("source") != expected_source
        or metadata.get("phase6_config_sha256") != canonical_text_sha256(config.config_path)
        or metadata.get("manifest_sha256") != sha256_file(config.artifacts.manifest)
    ):
        raise ConfigurationError("Mined-pair provenance does not match the locked Phase 6 config")
    return metadata


def _evaluate_validation(
    model: LearnedMultimodalFusion,
    loader: DataLoader[dict[str, Tensor | str]],
    split: EvaluationSplit,
    config: HardNegativeExperimentConfig,
    device: torch.device,
    *,
    minimum_recall: float,
) -> tuple[dict[str, Any], Ranking, Ranking]:
    posting_ids, _image, _text, joint, extraction_seconds = extract_joint_embeddings(
        model, loader, device
    )
    cosine_ranking, latency = rank_cosine_embeddings_profiled(
        posting_ids, joint, config.evaluation.candidate_k
    )
    pair_ranking = rerank_with_pair_head(model, posting_ids, joint, cosine_ranking, device)
    cosine_metrics = retrieval_metrics(
        cosine_ranking,
        split.label_by_id,
        config.evaluation.recall_at,
        config.evaluation.average_precision_at,
    )
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
                "retrieval": pair_metrics,
                "selected_pair_threshold": select_threshold(pair_ranking, split.label_by_id),
                "precision_at_controlled_recall": precision_at_minimum_recall(
                    pair_ranking, split.label_by_id, minimum_recall
                ),
            },
            "embedding_extraction_seconds": extraction_seconds,
            "embedding_throughput_per_second": len(posting_ids) / extraction_seconds,
            "search_latency": latency,
        },
        cosine_ranking,
        pair_ranking,
    )


def _failure_counts(
    cosine_ranking: Ranking,
    pair_ranking: Ranking,
    split: EvaluationSplit,
) -> dict[str, int]:
    item_by_id = {item.posting_id: item for item in split.items}
    top1_false = variant_conflict = regressions = retrieval_miss = 0
    for query_id, candidates in pair_ranking.items():
        label = split.label_by_id[query_id]
        pair_top = candidates[0].posting_id
        cosine_top = cosine_ranking[query_id][0].posting_id
        if split.label_by_id[pair_top] != label:
            top1_false += 1
            if has_variant_conflict(item_by_id[query_id].title, item_by_id[pair_top].title):
                variant_conflict += 1
            if split.label_by_id[cosine_top] == label:
                regressions += 1
        if not any(split.label_by_id[row.posting_id] == label for row in candidates):
            retrieval_miss += 1
    return {
        "false_top1": top1_false,
        "false_top1_variant_conflict": variant_conflict,
        "pair_head_regression": regressions,
        "retrieval_miss_at_20": retrieval_miss,
    }


def _checkpoint_payload(
    config: HardNegativeExperimentConfig,
    model: LearnedMultimodalFusion,
    optimizer: AdamW,
    scheduler: CosineAnnealingLR,
    *,
    epoch: int,
    best_metric: float,
    best_epoch: int,
    history: list[dict[str, Any]],
    manifest_sha256: str,
) -> dict[str, Any]:
    source = config.source.experiment
    return {
        "checkpoint_version": "phase6.hard_negative_finetune.v1",
        "epoch": epoch,
        "best_metric": best_metric,
        "best_epoch": best_epoch,
        "checkpoint_metric": config.evaluation.checkpoint_metric,
        "checkpoint_target": config.evaluation.checkpoint_target,
        "model_spec": asdict(source.model_spec),
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),  # type: ignore[no-untyped-call]
        "history": history,
        "seed": config.seed,
        "source_phase5_checkpoint_sha256": config.source.checkpoint_sha256,
        "hard_negative_manifest_sha256": manifest_sha256,
        "encoders_frozen": True,
    }


def _hard_pair_loss(
    model: LearnedMultimodalFusion,
    dataset: CachedMultimodalDataset,
    provider: HardNegativeBatchProvider,
    *,
    epoch: int,
    batch_index: int,
    count: int,
    device: torch.device,
) -> Tensor:
    left, right = provider.sample(epoch, batch_index, count)
    left_joint = model(
        dataset.image_embeddings[left].to(device),
        dataset.text_embeddings[left].to(device),
    )
    right_joint = model(
        dataset.image_embeddings[right].to(device),
        dataset.text_embeddings[right].to(device),
    )
    logits = model.pair_logits(left_joint, right_joint)
    return F.binary_cross_entropy_with_logits(logits, torch.zeros_like(logits))


def _render_report(run: dict[str, Any], mining_report: str) -> str:
    baseline = run["validation"]["phase5_baseline"]
    final = run["validation"]["selected_checkpoint"]
    baseline_pair = baseline["pair_head_rerank"]
    final_pair = final["pair_head_rerank"]
    history_rows = "\n".join(
        f"| {row['epoch'] + 1} | {row['train_loss']:.5f} | "
        f"{row['random_pair_bce']:.5f} | {row['hard_pair_bce']:.5f} | "
        f"{row['validation_map']:.5f} |"
        for row in run["history"]
    )
    checks = run["acceptance"]
    comparison_rows = "\n".join(
        (
            f"| Pair-head mAP@20 | {baseline_pair['retrieval']['map@20']:.5f} | "
            f"{final_pair['retrieval']['map@20']:.5f} | {checks['map_delta']:+.5f} |",
            f"| Pair-head Recall@20 | {baseline_pair['retrieval']['recall@20']:.5f} | "
            f"{final_pair['retrieval']['recall@20']:.5f} | "
            f"{checks['recall_at_20_delta']:+.5f} |",
            "| Precision at controlled recall | "
            f"{baseline_pair['precision_at_controlled_recall']['precision']:.5f} | "
            f"{final_pair['precision_at_controlled_recall']['precision']:.5f} | "
            f"{checks['controlled_precision_delta']:+.5f} |",
            "| False Top-1 variant conflicts | "
            f"{run['failures']['phase5_baseline']['false_top1_variant_conflict']} | "
            f"{run['failures']['selected_checkpoint']['false_top1_variant_conflict']} | "
            f"{checks['variant_conflict_delta']:+d} |",
        )
    )
    controlled_pass = str(checks["controlled_precision_pass"]).lower()
    if run["training"]["trainable_components"] == "pair_head":
        training_description = (
            "The fusion embedding was frozen after the joint pilot regressed. Only the pair head "
            "was optimized; supervised-contrastive loss was still measured as a diagnostic."
        )
    else:
        training_description = (
            "The fusion embedding and pair head were optimized jointly with supervised contrastive "
            "and mixed pair losses."
        )
    return f"""{mining_report.rstrip()}

## Fine-tuning result

The canonical Phase 5 weights initialized this run. Every optimization step retained the original
product-aware supervised-contrastive batch and random in-batch pair examples, then added a separate
batch of mined non-matches. The hard-negative BCE contributes only the configured share of the pair
loss; it does not replace the original training signal.

{training_description}

| Validation measure | Frozen Phase 5 | Phase 6 selected | Delta |
|---|---:|---:|---:|
{comparison_rows}

### Acceptance gates

- mAP non-regression: `{str(checks['map_pass']).lower()}`
- Recall@20 drop no greater than 0.002: `{str(checks['recall_pass']).lower()}`
- Precision improved at the Phase 5 recall target: `{controlled_pass}`
- Variant-conflict errors did not increase: `{str(checks['variant_conflict_pass']).lower()}`
- Pilot outcome: **{checks['status']}**

## Training history

| Epoch | Total loss | Random-pair BCE | Hard-negative BCE | Validation mAP@20 |
|---:|---:|---:|---:|---:|
{history_rows}

## Interpretation

`mAP@20` remains the checkpoint-selection metric because it measures ranking quality over all
queries. The controlled-recall precision is the Phase 6 diagnostic: it asks whether the system can
reject more look-alike non-matches while preserving the Phase 5 match-recall operating point. A
pilot is not considered closed evidence until its improvement is repeated across seeds.
"""


def run_hard_negative_experiment(
    config_path: Path, *, progress_updates_per_epoch: int = 4
) -> dict[str, object]:
    """Fine-tune the Phase 5 fusion model using train-only mined negative pairs."""
    if progress_updates_per_epoch < 0:
        raise ValueError("progress_updates_per_epoch must be non-negative")
    config = load_hard_negative_experiment_config(config_path)
    existing = [
        str(path)
        for path in (config.artifacts.checkpoint, config.artifacts.metrics)
        if path.exists()
    ]
    if existing:
        raise OutputConflictError(
            "Refusing to overwrite Phase 6 training evidence: " + ", ".join(existing)
        )
    metadata = _load_mining_metadata(config)
    pairs = load_hard_negative_manifest(config.artifacts.manifest)
    if len(pairs) != metadata["mined_pairs"]:
        raise DataValidationError("Mined manifest count differs from provenance metadata")
    seed_everything(config.seed, deterministic=config.training.deterministic)
    device = _resolve_device(config.training.device)
    source = config.source.experiment
    splits = load_splits(source.data.metadata_csv, source.data.split_manifest)
    train_dataset = load_cached_multimodal_split(source, "train")
    validation_dataset = load_cached_multimodal_split(source, "validation")
    sampler = ProductBatchSampler(
        train_dataset.labels,
        products_per_batch=config.training.products_per_batch,
        samples_per_product=config.training.samples_per_product,
        batches_per_epoch=config.training.batches_per_epoch,
        seed=config.seed,
    )
    provider = HardNegativeBatchProvider(
        pairs, train_dataset.posting_ids, train_dataset.labels, seed=config.seed
    )
    train_loader = DataLoader(train_dataset, batch_sampler=sampler, num_workers=0)
    validation_loader = DataLoader(validation_dataset, batch_size=512, shuffle=False, num_workers=0)
    model = load_phase5_source_model(config, device)
    if config.training.trainable_components == "pair_head":
        for parameter in model.fusion.parameters():
            parameter.requires_grad_(False)
    contrastive_loss = SupervisedContrastiveLoss(source.loss.temperature)
    trainable_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    if not trainable_parameters:
        raise ConfigurationError("Phase 6 config leaves no trainable model parameter")
    optimizer = AdamW(
        trainable_parameters,
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=config.training.epochs,
        eta_min=config.training.minimum_learning_rate,
    )
    source_pair = config.source.metrics["validation"]["selected_checkpoint"][
        "pair_head_rerank"
    ]
    target_recall = float(source_pair["selected_pair_threshold"]["recall"])
    baseline, baseline_cosine, baseline_pair = _evaluate_validation(
        model,
        validation_loader,
        splits["validation"],
        config,
        device,
        minimum_recall=target_recall,
    )
    source_map = float(source_pair["retrieval"]["map@20"])
    reproduced_map = float(baseline["pair_head_rerank"]["retrieval"]["map@20"])
    if not np.isclose(source_map, reproduced_map, rtol=0.0, atol=1e-12):
        raise DataValidationError("Phase 6 failed to reproduce the frozen Phase 5 validation mAP")
    manifest_sha = sha256_file(config.artifacts.manifest)
    best_metric = reproduced_map
    best_epoch = -1
    history: list[dict[str, Any]] = []
    _save_checkpoint_atomic(
        config.artifacts.checkpoint,
        _checkpoint_payload(
            config,
            model,
            optimizer,
            scheduler,
            epoch=-1,
            best_metric=best_metric,
            best_epoch=best_epoch,
            history=history,
            manifest_sha256=manifest_sha,
        ),
    )
    latest_path = config.artifacts.training_root / "latest.pt"
    LOGGER.info(
        "Phase 6 training: device=%s train=%d validation=%d mined=%d epochs=%d test=disabled",
        device,
        len(train_dataset),
        len(validation_dataset),
        len(pairs),
        config.training.epochs,
    )
    LOGGER.info(
        "frozen Phase 5 reproduction: map@20=%.5f controlled_precision=%.5f",
        reproduced_map,
        baseline["pair_head_rerank"]["precision_at_controlled_recall"]["precision"],
    )
    epochs_without_improvement = 0
    run_started = time.perf_counter()
    milestones = _progress_milestones(len(train_loader), progress_updates_per_epoch)
    for epoch in range(config.training.epochs):
        epoch_started = time.perf_counter()
        sampler.set_epoch(epoch)
        model.train()
        if config.training.trainable_components == "pair_head":
            model.fusion.eval()
        total_loss = total_supcon = total_random = total_hard = total_pair = 0.0
        LOGGER.info(
            "epoch %d/%d: mixed random/hard-negative training",
            epoch + 1,
            config.training.epochs,
        )
        for batch_index, raw_batch in enumerate(train_loader):
            batch = cast(dict[str, Any], raw_batch)
            image = _tensor_batch(batch, "image_embedding", device)
            text = _tensor_batch(batch, "text_embedding", device)
            labels = _tensor_batch(batch, "label", device)
            optimizer.zero_grad(set_to_none=True)
            joint = model(image, text)
            supcon = contrastive_loss(joint, labels)
            generator = torch.Generator(device="cpu")
            generator.manual_seed(config.seed + epoch * len(train_loader) + batch_index)
            left, right, targets = balanced_pair_indices(
                labels,
                maximum_negative_ratio=source.loss.maximum_negative_ratio,
                generator=generator,
            )
            random_pair = F.binary_cross_entropy_with_logits(
                model.pair_logits(joint[left], joint[right]), targets
            )
            hard_pair = _hard_pair_loss(
                model,
                train_dataset,
                provider,
                epoch=epoch,
                batch_index=batch_index,
                count=config.training.hard_pairs_per_batch,
                device=device,
            )
            fraction = config.training.hard_negative_loss_fraction
            pair_loss = (1 - fraction) * random_pair + fraction * hard_pair
            loss: Tensor
            if config.training.trainable_components == "pair_head":
                loss = pair_loss
            else:
                loss = (
                    source.loss.supervised_contrastive_weight * supcon
                    + source.loss.pair_bce_weight * pair_loss
                )
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite Phase 6 loss at epoch {epoch}")
            loss.backward()  # type: ignore[no-untyped-call]
            if epoch == 0 and batch_index == 0:
                _verify_finite_gradients(model)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.gradient_clip_norm)
            optimizer.step()
            total_loss += float(loss.detach().cpu())
            total_supcon += float(supcon.detach().cpu())
            total_random += float(random_pair.detach().cpu())
            total_hard += float(hard_pair.detach().cpu())
            total_pair += float(pair_loss.detach().cpu())
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
        model.eval()
        validation, _cosine, _pair = _evaluate_validation(
            model,
            validation_loader,
            splits["validation"],
            config,
            device,
            minimum_recall=target_recall,
        )
        current_metric = float(
            validation[config.evaluation.checkpoint_target]["retrieval"][
                config.evaluation.checkpoint_metric
            ]
        )
        record = {
            "epoch": epoch,
            "train_loss": total_loss / len(train_loader),
            "contrastive_loss": total_supcon / len(train_loader),
            "random_pair_bce": total_random / len(train_loader),
            "hard_pair_bce": total_hard / len(train_loader),
            "mixed_pair_bce": total_pair / len(train_loader),
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
            manifest_sha256=manifest_sha,
        )
        _save_checkpoint_atomic(latest_path, payload)
        if improved:
            _save_checkpoint_atomic(config.artifacts.checkpoint, payload)
        LOGGER.info(
            "epoch %d/%d complete: loss=%.5f map@20=%.5f best=%.5f elapsed=%s",
            epoch + 1,
            config.training.epochs,
            record["train_loss"],
            current_metric,
            best_metric,
            _format_duration(record["epoch_seconds"]),
        )
        if epochs_without_improvement >= config.training.early_stopping_patience:
            LOGGER.info("early stopping after %d unimproved epochs", epochs_without_improvement)
            break

    selected = torch.load(config.artifacts.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(selected["model_state"])
    selected_validation, selected_cosine, selected_pair = _evaluate_validation(
        model,
        validation_loader,
        splits["validation"],
        config,
        device,
        minimum_recall=target_recall,
    )
    baseline_failures = _failure_counts(
        baseline_cosine, baseline_pair, splits["validation"]
    )
    selected_failures = _failure_counts(
        selected_cosine, selected_pair, splits["validation"]
    )
    baseline_retrieval = baseline["pair_head_rerank"]["retrieval"]
    selected_retrieval = selected_validation["pair_head_rerank"]["retrieval"]
    baseline_controlled = baseline["pair_head_rerank"]["precision_at_controlled_recall"]
    selected_controlled = selected_validation["pair_head_rerank"][
        "precision_at_controlled_recall"
    ]
    map_delta = float(selected_retrieval["map@20"] - baseline_retrieval["map@20"])
    recall_delta = float(selected_retrieval["recall@20"] - baseline_retrieval["recall@20"])
    precision_delta = float(selected_controlled["precision"] - baseline_controlled["precision"])
    variant_delta = (
        selected_failures["false_top1_variant_conflict"]
        - baseline_failures["false_top1_variant_conflict"]
    )
    checks: dict[str, Any] = {
        "map_delta": map_delta,
        "recall_at_20_delta": recall_delta,
        "controlled_precision_delta": precision_delta,
        "variant_conflict_delta": variant_delta,
        "map_pass": map_delta >= -config.evaluation.maximum_map_drop,
        "recall_pass": recall_delta >= -config.evaluation.maximum_recall_at_20_drop,
        "controlled_precision_pass": precision_delta > 0,
        "variant_conflict_pass": variant_delta <= 0,
    }
    checks["pilot_pass"] = all(
        checks[name]
        for name in (
            "map_pass",
            "recall_pass",
            "controlled_precision_pass",
            "variant_conflict_pass",
        )
    )
    checks["status"] = (
        "pilot_pass_requires_repeated_seed_confirmation"
        if checks["pilot_pass"]
        else "pilot_did_not_pass"
    )
    commit, dirty = _git_state()
    run: dict[str, Any] = {
        "pipeline_version": "phase6.hard_negative_training.v1",
        "provenance": {
            "config_sha256": canonical_text_sha256(config.config_path),
            "source_checkpoint_sha256": config.source.checkpoint_sha256,
            "source_metrics_sha256": config.source.metrics_sha256,
            "manifest_sha256": manifest_sha,
            "git_commit": commit,
            "git_dirty": dirty,
            "seed": config.seed,
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "device": str(device),
        },
        "data": {
            "train": len(train_dataset),
            "validation": len(validation_dataset),
            "mined_pairs": len(pairs),
            "test_accessed": False,
        },
        "training": {
            "configured_epochs": config.training.epochs,
            "completed_epochs": len(history),
            "best_epoch": int(selected["best_epoch"]),
            "hard_pairs_per_batch": config.training.hard_pairs_per_batch,
            "hard_negative_loss_fraction": config.training.hard_negative_loss_fraction,
            "trainable_components": config.training.trainable_components,
            "supervised_contrastive_role": (
                "diagnostic_only"
                if config.training.trainable_components == "pair_head"
                else "optimized"
            ),
            "random_negatives_retained": True,
            "encoders_frozen": True,
        },
        "selection": {
            "split": "validation",
            "metric": config.evaluation.checkpoint_metric,
            "target": config.evaluation.checkpoint_target,
            "best_epoch": int(selected["best_epoch"]),
            "best_metric": float(selected["best_metric"]),
        },
        "history": history,
        "validation": {
            "controlled_recall_target": target_recall,
            "phase5_baseline": baseline,
            "selected_checkpoint": selected_validation,
        },
        "failures": {
            "phase5_baseline": baseline_failures,
            "selected_checkpoint": selected_failures,
        },
        "acceptance": checks,
        "test": {"status": "disabled_phase6_validation_only"},
        "efficiency": {"wall_time_seconds": time.perf_counter() - run_started},
    }
    _write_text_atomic(
        config.artifacts.metrics,
        json.dumps(run, indent=2, sort_keys=True) + "\n",
    )
    mining_report = config.artifacts.report.read_text(encoding="utf-8")
    _write_text_atomic(config.artifacts.report, _render_report(run, mining_report))
    return {
        "status": checks["status"],
        "checkpoint": str(config.artifacts.checkpoint),
        "metrics": str(config.artifacts.metrics),
        "report": str(config.artifacts.report),
        "best_epoch": int(selected["best_epoch"]),
        "validation_metric": float(selected["best_metric"]),
        "controlled_precision_delta": precision_delta,
        "test_status": run["test"]["status"],
    }
