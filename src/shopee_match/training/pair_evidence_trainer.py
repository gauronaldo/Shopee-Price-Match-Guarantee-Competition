"""Train a residual pair-evidence head without updating multimodal encoders."""

from __future__ import annotations

import itertools
import json
import logging
import math
import platform
import time
from collections import defaultdict
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from numpy.typing import NDArray
from torch.nn import functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset

from shopee_match.clustering.benchmark import (
    _failure_review,
    _write_assignments_atomic,
    _write_text_atomic,
)
from shopee_match.clustering.graph import ScoredPair, scored_pair_payload
from shopee_match.clustering.metrics import candidate_pair_classification_metrics, group_size_strata
from shopee_match.clustering.recall_recovery import _load_scored_pairs, sweep_recovery_policies
from shopee_match.errors import DataValidationError, OutputConflictError
from shopee_match.evaluation.protocol import EvaluationSplit, load_splits
from shopee_match.features.pair_evidence import (
    PAIR_EVIDENCE_FEATURES,
    PairEvidenceRecord,
    fit_pair_evidence_resources,
    pair_evidence_matrix,
    restore_pair_evidence_resources,
)
from shopee_match.features.text import CharTfidfModel
from shopee_match.hashing import canonical_text_sha256, sha256_file
from shopee_match.models import LearnedMultimodalFusion, ResidualPairEvidenceHead
from shopee_match.reproducibility import seed_everything
from shopee_match.retrieval.benchmark import load_phase6_model
from shopee_match.training.hard_negative_data import load_hard_negative_manifest
from shopee_match.training.multimodal_data import load_cached_multimodal_split
from shopee_match.training.multimodal_trainer import (
    _git_state,
    _resolve_device,
    _save_checkpoint_atomic,
    extract_joint_embeddings,
)
from shopee_match.training.pair_evidence_config import (
    PairEvidenceExperimentConfig,
    load_pair_evidence_experiment_config,
)

FloatArray = NDArray[np.float32]
IntArray = NDArray[np.int64]
LOGGER = logging.getLogger(__name__)


def _safe_logit(probabilities: FloatArray) -> FloatArray:
    clipped = np.clip(probabilities, 1e-6, 1 - 1e-6)
    return np.asarray(np.log(clipped / (1 - clipped)), dtype=np.float32)


def _score_index_pairs(
    model: LearnedMultimodalFusion,
    joint: FloatArray,
    indices: list[tuple[int, int]],
    device: torch.device,
    *,
    batch_size: int,
) -> tuple[FloatArray, FloatArray]:
    tensor = torch.from_numpy(joint)
    probabilities: list[float] = []
    cosine: list[float] = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(indices), batch_size):
            chunk = indices[start : start + batch_size]
            left_indices = [pair[0] for pair in chunk]
            right_indices = [pair[1] for pair in chunk]
            left = tensor[left_indices].to(device)
            right = tensor[right_indices].to(device)
            probabilities.extend(torch.sigmoid(model.pair_logits(left, right)).cpu().tolist())
            cosine.extend((left * right).sum(dim=1).cpu().tolist())
    return np.asarray(probabilities, dtype=np.float32), np.asarray(cosine, dtype=np.float32)


def _positive_indices(labels: tuple[str, ...]) -> list[tuple[int, int]]:
    members: dict[str, list[int]] = defaultdict(list)
    for index, label in enumerate(labels):
        members[label].append(index)
    return [pair for label in sorted(members) for pair in itertools.combinations(members[label], 2)]


def _random_negative_indices(
    labels: tuple[str, ...],
    excluded: set[tuple[int, int]],
    count: int,
    *,
    seed: int,
) -> list[tuple[int, int]]:
    generator = np.random.default_rng(seed)
    selected: set[tuple[int, int]] = set()
    while len(selected) < count:
        left = int(generator.integers(0, len(labels)))
        right = int(generator.integers(0, len(labels)))
        if left == right or labels[left] == labels[right]:
            continue
        pair = (left, right) if left < right else (right, left)
        if pair not in excluded:
            selected.add(pair)
    return sorted(selected)


def _pair_records(
    posting_ids: tuple[str, ...],
    indices: list[tuple[int, int]],
    probabilities: FloatArray,
    cosine: FloatArray,
) -> list[PairEvidenceRecord]:
    return [
        PairEvidenceRecord(
            posting_ids[left],
            posting_ids[right],
            float(probability),
            float(similarity),
        )
        for (left, right), probability, similarity in zip(
            indices, probabilities, cosine, strict=True
        )
    ]


def _rescore_pairs(
    head: ResidualPairEvidenceHead,
    pairs: list[ScoredPair],
    evidence: FloatArray,
    device: torch.device,
    *,
    batch_size: int,
) -> list[ScoredPair]:
    baseline = _safe_logit(np.asarray([pair.pair_probability for pair in pairs], dtype=np.float32))
    probabilities: list[float] = []
    head.eval()
    with torch.inference_mode():
        for start in range(0, len(pairs), batch_size):
            stop = min(start + batch_size, len(pairs))
            logits = head(
                torch.from_numpy(baseline[start:stop]).to(device),
                torch.from_numpy(evidence[start:stop]).to(device),
            )
            probabilities.extend(torch.sigmoid(logits).cpu().tolist())
    return [
        replace(pair, pair_probability=float(probability))
        for pair, probability in zip(pairs, probabilities, strict=True)
    ]


def _pair_metrics(pairs: list[ScoredPair], split: EvaluationSplit) -> dict[str, float]:
    return candidate_pair_classification_metrics(
        pairs,
        split.label_by_id,
        threshold=0.5,
        calibration_bins=10,
        required_recall=0.8,
        required_precision=0.9,
    )


def _checkpoint_payload(
    config: PairEvidenceExperimentConfig,
    head: ResidualPairEvidenceHead,
    tfidf: CharTfidfModel,
    *,
    epoch: int,
    best_metric: float,
    history: list[dict[str, float]],
) -> dict[str, Any]:
    return {
        "checkpoint_version": "pair_evidence.training.v1",
        "epoch": epoch,
        "best_metric": best_metric,
        "metric": "validation_candidate_pair_average_precision",
        "feature_names": PAIR_EVIDENCE_FEATURES,
        "model_state": head.state_dict(),
        "tfidf": {
            "vocabulary": tfidf.vocabulary,
            "idf": tfidf.idf,
            "ngram_range": tfidf.ngram_range,
        },
        "history": history,
        "seed": config.seed,
        "recovery_config_sha256": config.recovery_config_sha256,
        "encoders_frozen": True,
        "fusion_frozen": True,
        "phase6_pair_head_frozen": True,
    }


def _render_report(run: dict[str, Any]) -> str:
    baseline = run["validation"]["baseline_pair_metrics"]
    evidence = run["validation"]["evidence_pair_metrics"]
    selected = run["graph_selection"]["selected"]
    cluster = selected["clustering"]
    pairwise = cluster["pairwise"]
    average_precision_row = (
        f"| Average precision / PR-AUC | {baseline['average_precision_pr_auc']:.5f} | "
        f"{evidence['average_precision_pr_auc']:.5f} |"
    )
    calibration_row = (
        f"| Expected calibration error | {baseline['expected_calibration_error']:.5f} | "
        f"{evidence['expected_calibration_error']:.5f} |"
    )
    return f"""# Pair-Evidence Recall Recovery

Status: **{run["status"]}**. The accepted image, text, fusion, and Phase 6 pair-head parameters
remain frozen. Only an `{run["model"]["parameters"]}`-parameter residual evidence head is trained.

## Candidate-pair ranking

| Validation metric | Frozen pair head | Evidence head |
|---|---:|---:|
{average_precision_row}
| Brier score | {baseline["brier_score"]:.5f} | {evidence["brier_score"]:.5f} |
{calibration_row}

## Selected validation graph

| Metric | Value |
|---|---:|
| Pairwise precision | {pairwise["precision"]:.5f} |
| Pairwise recall | {pairwise["recall"]:.5f} |
| Pairwise F1 | {pairwise["f1"]:.5f} |
| B-cubed F1 | {cluster["b_cubed"]["f1"]:.5f} |
| False-merge pair rate | {cluster["false_merge_pair_rate"]:.5f} |
| False-split group rate | {cluster["false_split_group_rate"]:.5f} |

The evidence vector is label-blind at inference: frozen pair probability, joint cosine similarity,
pHash similarity, train-fitted character TF-IDF similarity, title-token overlap, digit/unit overlap,
variant conflict, exact-title/pHash flags, and title-length ratio. Labels are used only to sample
train pairs and evaluate validation policies. Test remains untouched.

## Reproduction

```powershell
.venv\\Scripts\\shopee-entity-resolution train-pair-evidence `
  --config configs\\experiment\\pair_evidence_training.yaml
```
"""


def _prepare_training_pairs(
    config: PairEvidenceExperimentConfig,
    model: LearnedMultimodalFusion,
    train_split: EvaluationSplit,
    posting_ids: tuple[str, ...],
    joint: FloatArray,
    device: torch.device,
) -> tuple[list[PairEvidenceRecord], FloatArray, FloatArray, dict[str, int]]:
    index_by_id = {posting_id: index for index, posting_id in enumerate(posting_ids)}
    labels = tuple(train_split.label_by_id[posting_id] for posting_id in posting_ids)
    positives = _positive_indices(labels)
    phase7 = config.recovery.source.experiment.source.experiment
    hard_manifest = load_hard_negative_manifest(phase7.source.mined_manifest_path)
    hard_negative_set: set[tuple[int, int]] = set()
    for pair in hard_manifest:
        left = index_by_id[pair.left_posting_id]
        right = index_by_id[pair.right_posting_id]
        hard_negative_set.add((left, right) if left < right else (right, left))
    hard_negatives = sorted(hard_negative_set)
    target_negatives = math.ceil(len(positives) * config.training.negative_to_positive_ratio)
    if target_negatives < len(hard_negatives):
        hard_negatives = hard_negatives[:target_negatives]
    random_count = target_negatives - len(hard_negatives)
    random_negatives = _random_negative_indices(
        labels,
        set(hard_negatives),
        random_count,
        seed=config.seed,
    )
    indices = positives + hard_negatives + random_negatives
    probability, cosine = _score_index_pairs(
        model,
        joint,
        indices,
        device,
        batch_size=config.training.batch_size,
    )
    targets = np.concatenate(
        (
            np.ones(len(positives), dtype=np.float32),
            np.zeros(len(hard_negatives) + len(random_negatives), dtype=np.float32),
        )
    )
    weights = np.ones(len(indices), dtype=np.float32)
    cutoff = float(
        np.quantile(probability[: len(positives)], config.training.hard_positive_fraction)
    )
    hard_positive_mask = probability[: len(positives)] <= cutoff
    weights[: len(positives)][hard_positive_mask] = config.training.hard_positive_weight
    hard_start, hard_stop = len(positives), len(positives) + len(hard_negatives)
    weights[hard_start:hard_stop] = config.training.hard_negative_weight
    return (
        _pair_records(posting_ids, indices, probability, cosine),
        targets,
        weights,
        {
            "positive_pairs": len(positives),
            "hard_positive_pairs": int(hard_positive_mask.sum()),
            "hard_negative_pairs": len(hard_negatives),
            "random_negative_pairs": len(random_negatives),
        },
    )


def run_pair_evidence_experiment(config_path: Path) -> dict[str, object]:
    """Train on train-only pair evidence and select the graph policy on validation."""
    config = load_pair_evidence_experiment_config(config_path)
    outputs = (
        config.artifacts.checkpoint,
        config.artifacts.rescored_pairs,
        config.artifacts.assignments,
        config.artifacts.metrics,
        config.artifacts.review,
        config.artifacts.report,
    )
    existing = [str(path) for path in outputs if path.exists()]
    if existing:
        raise OutputConflictError(
            "Refusing to overwrite pair-evidence evidence: " + ", ".join(existing)
        )
    started = time.perf_counter()
    seed_everything(config.seed, deterministic=config.training.deterministic)
    device = _resolve_device(config.training.device)
    recovery = config.recovery
    entity = recovery.source.experiment
    phase7 = entity.source.experiment
    phase6 = phase7.source.experiment
    phase5 = phase6.source.experiment
    splits = load_splits(phase5.data.metadata_csv, phase5.data.split_manifest)
    model = load_phase6_model(phase7, device)
    train_dataset = load_cached_multimodal_split(phase5, "train")
    train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=False, num_workers=0)
    LOGGER.info("Pair evidence stage 1/5: extracting frozen train joint embeddings on %s", device)
    train_ids, _image, _text, train_joint, _seconds = extract_joint_embeddings(
        model, train_loader, device
    )
    if train_ids != tuple(item.posting_id for item in splits["train"].items):
        raise DataValidationError("Train cache does not align with the split manifest")
    LOGGER.info("Pair evidence stage 2/5: mining all train positives and balanced negatives")
    records, targets, weights, pair_counts = _prepare_training_pairs(
        config,
        model,
        splits["train"],
        train_ids,
        train_joint,
        device,
    )
    LOGGER.info(
        "Training pairs: positives=%d hard_positives=%d hard_negatives=%d random_negatives=%d",
        pair_counts["positive_pairs"],
        pair_counts["hard_positive_pairs"],
        pair_counts["hard_negative_pairs"],
        pair_counts["random_negative_pairs"],
    )
    resources = fit_pair_evidence_resources(
        splits["train"].items,
        splits["train"].items,
        ngram_range=config.tfidf.ngram_range,
        max_features=config.tfidf.max_features,
    )
    evidence = pair_evidence_matrix(records, resources)
    feature_mean = evidence.mean(axis=0, dtype=np.float64).astype(np.float32)
    feature_scale = evidence.std(axis=0, dtype=np.float64).astype(np.float32)
    feature_scale[feature_scale < 1e-6] = 1.0

    validation_ids = tuple(item.posting_id for item in splits["validation"].items)
    validation_pairs = _load_scored_pairs(recovery, validation_ids)
    validation_resources = restore_pair_evidence_resources(
        resources.tfidf, splits["validation"].items
    )
    validation_records = [
        PairEvidenceRecord(
            pair.left_posting_id,
            pair.right_posting_id,
            pair.pair_probability,
            pair.cosine_similarity,
        )
        for pair in validation_pairs
    ]
    validation_evidence = pair_evidence_matrix(validation_records, validation_resources)
    baseline_pair_metrics = _pair_metrics(validation_pairs, splits["validation"])

    head = ResidualPairEvidenceHead(
        torch.from_numpy(feature_mean), torch.from_numpy(feature_scale)
    ).to(device)
    optimizer = AdamW(
        head.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    baseline_logits = _safe_logit(
        np.asarray([record.baseline_probability for record in records], dtype=np.float32)
    )
    dataset = TensorDataset(
        torch.from_numpy(baseline_logits),
        torch.from_numpy(evidence),
        torch.from_numpy(targets),
        torch.from_numpy(weights),
    )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(config.seed)
    loader = DataLoader(
        dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
    )
    baseline_metric = float(baseline_pair_metrics["average_precision_pr_auc"])
    best_metric = baseline_metric
    best_epoch = -1
    history: list[dict[str, float]] = []
    _save_checkpoint_atomic(
        config.artifacts.checkpoint,
        _checkpoint_payload(
            config,
            head,
            resources.tfidf,
            epoch=-1,
            best_metric=best_metric,
            history=history,
        ),
    )
    LOGGER.info(
        "Pair evidence stage 3/5: training %d parameters for at most %d epochs",
        head.parameter_count,
        config.training.epochs,
    )
    epochs_without_improvement = 0
    for epoch in range(config.training.epochs):
        head.train()
        total_loss = 0.0
        for raw_baseline, raw_evidence, raw_targets, raw_weights in loader:
            optimizer.zero_grad(set_to_none=True)
            logits = head(raw_baseline.to(device), raw_evidence.to(device))
            losses = F.binary_cross_entropy_with_logits(
                logits, raw_targets.to(device), reduction="none"
            )
            loss = (losses * raw_weights.to(device)).mean()
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite pair-evidence loss")
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().cpu())
        rescored = _rescore_pairs(
            head,
            validation_pairs,
            validation_evidence,
            device,
            batch_size=config.training.batch_size,
        )
        metrics = _pair_metrics(rescored, splits["validation"])
        current = float(metrics["average_precision_pr_auc"])
        history.append(
            {
                "epoch": float(epoch),
                "train_loss": total_loss / len(loader),
                "validation_average_precision": current,
                "validation_brier_score": float(metrics["brier_score"]),
            }
        )
        if current > best_metric:
            best_metric, best_epoch, epochs_without_improvement = current, epoch, 0
            _save_checkpoint_atomic(
                config.artifacts.checkpoint,
                _checkpoint_payload(
                    config,
                    head,
                    resources.tfidf,
                    epoch=epoch,
                    best_metric=best_metric,
                    history=history,
                ),
            )
        else:
            epochs_without_improvement += 1
        LOGGER.info(
            "epoch %d/%d: loss=%.5f validation_ap=%.5f best=%.5f",
            epoch + 1,
            config.training.epochs,
            total_loss / len(loader),
            current,
            best_metric,
        )
        if epochs_without_improvement >= config.training.early_stopping_patience:
            LOGGER.info(
                "Pair-evidence early stopping after %d unimproved epochs",
                epochs_without_improvement,
            )
            break

    checkpoint = torch.load(config.artifacts.checkpoint, map_location=device, weights_only=False)
    head.load_state_dict(checkpoint["model_state"])
    rescored_pairs = _rescore_pairs(
        head,
        validation_pairs,
        validation_evidence,
        device,
        batch_size=config.training.batch_size,
    )
    evidence_pair_metrics = _pair_metrics(rescored_pairs, splits["validation"])
    LOGGER.info("Pair evidence stage 4/5: selecting graph policy from rescored validation pairs")
    trials, selected_trial, assignments = sweep_recovery_policies(
        recovery,
        validation_ids,
        rescored_pairs,
        splits["validation"].label_by_id,
    )
    review = _failure_review(
        splits["validation"],
        assignments,
        example_limit=recovery.selection.failure_example_limit,
    )
    status = (
        "pair_evidence_target_reached_validation_only"
        if selected_trial["passes_quality_target"]
        else "pair_evidence_target_not_reached_validation_only"
    )
    commit, dirty = _git_state()
    run: dict[str, Any] = {
        "pipeline_version": "pair_evidence.training.v1",
        "status": status,
        "provenance": {
            "config_sha256": canonical_text_sha256(config.config_path),
            "recovery_config_sha256": config.recovery_config_sha256,
            "entity_metrics_sha256": recovery.source.entity_metrics_sha256,
            "source_scored_pairs_sha256": recovery.source.scored_pairs_sha256,
            "phase6_checkpoint_sha256": phase7.source.checkpoint_sha256,
            "hard_negative_manifest_sha256": phase7.source.mined_manifest_sha256,
            "split_manifest_sha256": sha256_file(phase5.data.split_manifest),
            "git_commit": commit,
            "git_dirty": dirty,
            "seed": config.seed,
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "device": str(device),
        },
        "data": {
            "fit_split": "train",
            "tune_split": "validation",
            "train_listings": len(train_ids),
            "validation_listings": len(validation_ids),
            "test_accessed": False,
            **pair_counts,
        },
        "model": {
            "name": "residual_pair_evidence_head",
            "parameters": head.parameter_count,
            "feature_names": PAIR_EVIDENCE_FEATURES,
            "encoders_frozen": True,
            "fusion_frozen": True,
            "phase6_pair_head_frozen": True,
        },
        "training": {
            "best_epoch": int(checkpoint["epoch"]),
            "best_validation_average_precision": float(checkpoint["best_metric"]),
            "history": history,
        },
        "validation": {
            "baseline_pair_metrics": baseline_pair_metrics,
            "evidence_pair_metrics": evidence_pair_metrics,
        },
        "graph_selection": {
            "acceptance": asdict(recovery.selection.acceptance),
            "selected": selected_trial,
            "trials": trials,
            "group_size_strata": group_size_strata(assignments, splits["validation"].label_by_id),
        },
        "failure_analysis": review["counts"],
        "runtime_seconds": time.perf_counter() - started,
        "test": {"status": "disabled_pair_evidence_validation_only"},
    }
    LOGGER.info("Pair evidence stage 5/5: writing immutable local evidence")
    _write_text_atomic(
        config.artifacts.rescored_pairs,
        "".join(
            json.dumps(scored_pair_payload(pair), sort_keys=True) + "\n" for pair in rescored_pairs
        ),
    )
    _write_assignments_atomic(config.artifacts.assignments, assignments)
    _write_text_atomic(config.artifacts.review, json.dumps(review, indent=2, sort_keys=True) + "\n")
    _write_text_atomic(config.artifacts.metrics, json.dumps(run, indent=2, sort_keys=True) + "\n")
    _write_text_atomic(config.artifacts.report, _render_report(run))
    cluster = cast(dict[str, Any], selected_trial["clustering"])
    return {
        "status": status,
        "best_epoch": best_epoch,
        "validation_pair_average_precision": evidence_pair_metrics["average_precision_pr_auc"],
        "pairwise_precision": cluster["pairwise"]["precision"],
        "pairwise_recall": cluster["pairwise"]["recall"],
        "pairwise_f1": cluster["pairwise"]["f1"],
        "b_cubed_f1": cluster["b_cubed"]["f1"],
        "false_merge_pair_rate": cluster["false_merge_pair_rate"],
        "false_split_group_rate": cluster["false_split_group_rate"],
        "checkpoint": str(config.artifacts.checkpoint),
        "metrics": str(config.artifacts.metrics),
        "report": str(config.artifacts.report),
        "test_accessed": False,
    }
