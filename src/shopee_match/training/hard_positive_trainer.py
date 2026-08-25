"""Train-only hard-positive mining and pair-head fine-tuning."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from numpy.typing import NDArray
from sklearn.metrics import average_precision_score  # type: ignore[import-untyped]
from torch import Tensor
from torch.optim import AdamW

from shopee_match.clustering.graph import score_candidate_pairs
from shopee_match.clustering.hybrid_entity import _load_ranking
from shopee_match.clustering.recall_recovery import sweep_recovery_policies
from shopee_match.errors import DataValidationError, OutputConflictError
from shopee_match.evaluation.protocol import load_splits
from shopee_match.hashing import canonical_text_sha256, sha256_file
from shopee_match.models.multimodal_fusion import LearnedMultimodalFusion
from shopee_match.reproducibility import seed_everything
from shopee_match.retrieval.benchmark import load_phase6_model
from shopee_match.retrieval.hybrid_benchmark import _load_embeddings
from shopee_match.training.hard_negative_data import load_hard_negative_manifest
from shopee_match.training.hard_positive_config import (
    HardPositiveExperimentConfig,
    HardPositiveTrainingConfig,
    load_hard_positive_experiment_config,
)
from shopee_match.training.multimodal_data import (
    CachedMultimodalDataset,
    load_cached_multimodal_split,
)
from shopee_match.training.multimodal_trainer import _git_state, _resolve_device

LOGGER = logging.getLogger(__name__)
FloatArray = NDArray[np.float32]
IndexPair = tuple[int, int]


@dataclass(frozen=True, slots=True)
class GroupPartition:
    optimization_indices: tuple[int, ...]
    holdout_indices: tuple[int, ...]
    optimization_groups: int
    holdout_groups: int


def deterministic_group_holdout(
    labels: tuple[str, ...], *, holdout_fraction: float, seed: int
) -> GroupPartition:
    """Create a deterministic group-disjoint holdout stratified coarsely by group size."""
    if not 0.0 < holdout_fraction < 1.0:
        raise ValueError("holdout_fraction must be inside (0, 1)")
    members_by_label: dict[str, list[int]] = defaultdict(list)
    for index, label in enumerate(labels):
        members_by_label[label].append(index)
    strata: dict[str, list[str]] = defaultdict(list)
    for label, members in members_by_label.items():
        size = len(members)
        stratum = "2" if size == 2 else "3_to_5" if size <= 5 else "6_plus"
        strata[stratum].append(label)
    holdout_labels: set[str] = set()
    for stratum in sorted(strata):
        ordered = sorted(
            strata[stratum],
            key=lambda label: hashlib.sha256(f"{seed}:{label}".encode()).hexdigest(),
        )
        count = min(len(ordered) - 1, max(1, round(len(ordered) * holdout_fraction)))
        holdout_labels.update(ordered[:count])
    optimization = tuple(
        index for index, label in enumerate(labels) if label not in holdout_labels
    )
    holdout = tuple(index for index, label in enumerate(labels) if label in holdout_labels)
    if not optimization or not holdout:
        raise DataValidationError("Hard-positive group partition produced an empty side")
    return GroupPartition(
        optimization,
        holdout,
        len(members_by_label) - len(holdout_labels),
        len(holdout_labels),
    )


def positive_pairs(labels: tuple[str, ...], indices: tuple[int, ...]) -> list[IndexPair]:
    """Return sorted same-group pairs inside an allowed index set."""
    by_label: dict[str, list[int]] = defaultdict(list)
    for index in indices:
        by_label[labels[index]].append(index)
    return sorted(
        pair
        for members in by_label.values()
        for pair in combinations(sorted(members), 2)
    )


def _score_index_pairs(
    model: LearnedMultimodalFusion,
    embeddings: Tensor,
    pairs: list[IndexPair],
    *,
    batch_size: int,
    device: torch.device,
) -> FloatArray:
    probabilities: list[Tensor] = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(pairs), batch_size):
            chunk = pairs[start : start + batch_size]
            left = embeddings[[pair[0] for pair in chunk]].to(device)
            right = embeddings[[pair[1] for pair in chunk]].to(device)
            probabilities.append(torch.sigmoid(model.pair_logits(left, right)).cpu())
    if not probabilities:
        return np.empty(0, dtype=np.float32)
    return torch.cat(probabilities).numpy().astype(np.float32, copy=False)


def select_hard_positive_pairs(
    pairs: list[IndexPair], probabilities: FloatArray, *, limit: int
) -> list[IndexPair]:
    """Select the lowest-scoring true pairs with deterministic tie-breaking."""
    if len(pairs) != len(probabilities) or limit <= 0:
        raise ValueError("hard-positive pairs, probabilities, and limit are inconsistent")
    order = sorted(range(len(pairs)), key=lambda index: (probabilities[index], pairs[index]))
    return [pairs[index] for index in order[:limit]]


class MixedPairBatchProvider:
    """Sample hard and random positive/negative pairs deterministically."""

    def __init__(
        self,
        *,
        hard_positives: list[IndexPair],
        hard_negatives: list[IndexPair],
        random_positives: list[IndexPair],
        allowed_indices: tuple[int, ...],
        labels: tuple[str, ...],
        seed: int,
    ) -> None:
        if not hard_positives or not hard_negatives or not random_positives:
            raise DataValidationError("Mixed pair training requires all three pair pools")
        allowed = set(allowed_indices)
        if any(left not in allowed or right not in allowed for left, right in hard_positives):
            raise DataValidationError("Hard-positive pair crosses the optimization partition")
        if any(left not in allowed or right not in allowed for left, right in hard_negatives):
            raise DataValidationError("Hard-negative pair crosses the optimization partition")
        self.hard_positives = tuple(hard_positives)
        self.hard_negatives = tuple(hard_negatives)
        self.random_positives = tuple(random_positives)
        self.allowed_indices = allowed_indices
        self.labels = labels
        self.seed = seed

    @staticmethod
    def _counts(batch_size: int, config: HardPositiveTrainingConfig) -> tuple[int, int, int, int]:
        hard_positive = round(batch_size * config.hard_positive_fraction)
        hard_negative = round(batch_size * config.hard_negative_fraction)
        random_positive = round(batch_size * config.random_positive_fraction)
        random_negative = batch_size - hard_positive - hard_negative - random_positive
        if min(hard_positive, hard_negative, random_positive, random_negative) <= 0:
            raise ValueError("batch size is too small for configured pair fractions")
        return hard_positive, hard_negative, random_positive, random_negative

    def sample(
        self,
        *,
        epoch: int,
        batch_index: int,
        batch_size: int,
        config: HardPositiveTrainingConfig,
    ) -> tuple[Tensor, Tensor, Tensor]:
        rng = random.Random(f"{self.seed}:{epoch}:{batch_index}")
        counts = self._counts(batch_size, config)
        pairs = (
            rng.choices(self.hard_positives, k=counts[0])
            + rng.choices(self.hard_negatives, k=counts[1])
            + rng.choices(self.random_positives, k=counts[2])
        )
        targets = [1.0] * counts[0] + [0.0] * counts[1] + [1.0] * counts[2]
        for _ in range(counts[3]):
            left = rng.choice(self.allowed_indices)
            right = rng.choice(self.allowed_indices)
            while left == right or self.labels[left] == self.labels[right]:
                right = rng.choice(self.allowed_indices)
            pairs.append((left, right))
            targets.append(0.0)
        order = list(range(batch_size))
        rng.shuffle(order)
        return (
            torch.tensor([pairs[index][0] for index in order], dtype=torch.long),
            torch.tensor([pairs[index][1] for index in order], dtype=torch.long),
            torch.tensor([targets[index] for index in order], dtype=torch.float32),
        )


def _extract_joint_embeddings(
    model: LearnedMultimodalFusion,
    dataset: CachedMultimodalDataset,
    device: torch.device,
    *,
    batch_size: int,
) -> Tensor:
    arrays: list[Tensor] = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(dataset), batch_size):
            stop = start + batch_size
            arrays.append(
                model(
                    dataset.image_embeddings[start:stop].to(device),
                    dataset.text_embeddings[start:stop].to(device),
                ).cpu()
            )
    return torch.cat(arrays)


def _holdout_candidate_pairs(
    embeddings: Tensor,
    labels: tuple[str, ...],
    holdout_indices: tuple[int, ...],
    *,
    candidate_k: int,
) -> tuple[list[IndexPair], NDArray[np.int64]]:
    local = embeddings[list(holdout_indices)]
    k = min(candidate_k, len(local) - 1)
    similarity = local @ local.T
    similarity.fill_diagonal_(-torch.inf)
    neighbours = torch.topk(similarity, k=k, dim=1, sorted=True).indices.cpu().tolist()
    candidates: set[IndexPair] = set()
    for query_local, rows in enumerate(neighbours):
        query = holdout_indices[query_local]
        for candidate_local in rows:
            candidate = holdout_indices[candidate_local]
            candidates.add((query, candidate) if query < candidate else (candidate, query))
    true_pairs = positive_pairs(labels, holdout_indices)
    candidates.update(true_pairs)
    ordered = sorted(candidates)
    targets = np.asarray([labels[left] == labels[right] for left, right in ordered], dtype=np.int64)
    return ordered, targets


def _pair_metrics(
    targets: NDArray[np.int64], probabilities: FloatArray, *, minimum_precision: float
) -> dict[str, float]:
    if len(targets) != len(probabilities) or not np.any(targets == 1):
        raise DataValidationError("Holdout pair metrics require aligned positive examples")
    order = np.argsort(-probabilities, kind="stable")
    sorted_targets = targets[order]
    true_positive = np.cumsum(sorted_targets)
    predicted = np.arange(1, len(targets) + 1)
    precision = true_positive / predicted
    recall = true_positive / int(targets.sum())
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(precision, dtype=np.float64),
        where=(precision + recall) > 0,
    )
    eligible = np.flatnonzero(precision >= minimum_precision)
    selected = int(eligible[np.argmax(f1[eligible])]) if len(eligible) else -1
    if selected < 0:
        threshold, selected_precision, selected_recall, selected_f1 = 1.0, 1.0, 0.0, 0.0
    else:
        threshold = float(probabilities[order[selected]])
        selected_precision = float(precision[selected])
        selected_recall = float(recall[selected])
        selected_f1 = float(f1[selected])
    return {
        "average_precision": float(average_precision_score(targets, probabilities)),
        "threshold": threshold,
        "precision": selected_precision,
        "recall": selected_recall,
        "f1": selected_f1,
        "pairs": float(len(targets)),
        "positive_pairs": float(targets.sum()),
    }


def _validation_graph(
    model: LearnedMultimodalFusion,
    config: HardPositiveExperimentConfig,
    device: torch.device,
) -> dict[str, Any]:
    hybrid = config.source.hybrid_entity.source.hybrid
    phase7 = hybrid.source.experiment
    phase6 = phase7.source.experiment
    phase5 = phase6.source.experiment
    split = load_splits(phase5.data.metadata_csv, phase5.data.split_manifest)["validation"]
    posting_ids, embeddings = _load_embeddings(hybrid.source.embedding_cache_path)
    ranking = _load_ranking(
        config.source.hybrid_entity.source.hybrid_ranking_path,
        posting_ids,
        expected_k=int(config.source.hybrid_entity.source.hybrid_metrics["selection"]["candidate_k"]),
    )
    pairs = score_candidate_pairs(
        model,
        posting_ids,
        split.items,
        embeddings,
        ranking,
        device,
        batch_size=config.mining.scoring_batch_size,
    )
    trials, selected, _assignments = sweep_recovery_policies(
        config.source.recovery,
        posting_ids,
        pairs,
        split.label_by_id,
    )
    return {"scored_pairs": len(pairs), "selected": selected, "trials": len(trials)}


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


def _render_report(run: dict[str, Any]) -> str:
    baseline = run["validation_graph"]["baseline"]["clustering"]
    final = run["validation_graph"]["selected_checkpoint"]["clustering"]
    rows = "\n".join(
        (
            f"| Pairwise precision | {baseline['pairwise']['precision']:.5f} | "
            f"{final['pairwise']['precision']:.5f} |",
            f"| Pairwise recall | {baseline['pairwise']['recall']:.5f} | "
            f"{final['pairwise']['recall']:.5f} |",
            f"| Pairwise F1 | {baseline['pairwise']['f1']:.5f} | "
            f"{final['pairwise']['f1']:.5f} |",
            f"| B-cubed F1 | {baseline['b_cubed']['f1']:.5f} | "
            f"{final['b_cubed']['f1']:.5f} |",
            f"| False-merge pair rate | {baseline['false_merge_pair_rate']:.5f} | "
            f"{final['false_merge_pair_rate']:.5f} |",
            f"| False-split group rate | {baseline['false_split_group_rate']:.5f} | "
            f"{final['false_split_group_rate']:.5f} |",
        )
    )
    return f"""# Hard-positive Pair-Head Fine-Tuning

This train-only experiment fine-tunes the symmetric pair head while keeping both encoders and the
multimodal fusion module frozen. Checkpoint selection uses a new group-disjoint holdout carved from
the original train partition. Test data is not accessed.

| Validation graph metric | Before | After |
|---|---:|---:|
{rows}

Status: **{run['status']}**. The original frozen test policy is unchanged.
"""


def run_hard_positive_experiment(config_path: Path) -> dict[str, object]:
    """Mine hard positives from train, fine-tune the pair head, and evaluate validation once."""
    config = load_hard_positive_experiment_config(config_path)
    existing = [
        str(path)
        for path in (config.artifacts.checkpoint, config.artifacts.metrics, config.artifacts.report)
        if path.exists()
    ]
    if existing:
        details = ", ".join(existing)
        raise OutputConflictError(f"Refusing to overwrite hard-positive evidence: {details}")
    started = time.perf_counter()
    seed_everything(config.seed, deterministic=True)
    device = _resolve_device(config.training.device)
    model = load_phase6_model(config.source.candidate, device)
    multimodal = config.source.candidate.source.experiment.source.experiment
    dataset = load_cached_multimodal_split(multimodal, "train")
    splits = load_splits(multimodal.data.metadata_csv, multimodal.data.split_manifest)
    split = splits["train"]
    if dataset.posting_ids != tuple(item.posting_id for item in split.items):
        raise DataValidationError("Train cache and split order differ")
    partition = deterministic_group_holdout(
        dataset.labels,
        holdout_fraction=config.data.holdout_fraction,
        seed=config.data.split_seed,
    )
    LOGGER.info(
        "Hard-positive stage 1/5: extracting frozen joint embeddings on %s", device
    )
    joint = _extract_joint_embeddings(
        model,
        dataset,
        device,
        batch_size=config.mining.scoring_batch_size,
    )
    all_optimization_positives = positive_pairs(dataset.labels, partition.optimization_indices)
    item_by_id = {item.posting_id: item for item in split.items}
    if config.mining.require_different_phash:
        all_optimization_positives = [
            pair
            for pair in all_optimization_positives
            if item_by_id[dataset.posting_ids[pair[0]]].image_phash
            != item_by_id[dataset.posting_ids[pair[1]]].image_phash
        ]
    if not all_optimization_positives:
        raise DataValidationError(
            "No eligible train-only positive pairs remain after hard-positive filtering"
        )
    LOGGER.info(
        "Hard-positive stage 2/5: scoring %d train-only positive pairs",
        len(all_optimization_positives),
    )
    positive_probabilities = _score_index_pairs(
        model,
        joint,
        all_optimization_positives,
        batch_size=config.mining.scoring_batch_size,
        device=device,
    )
    hard_positives = select_hard_positive_pairs(
        all_optimization_positives,
        positive_probabilities,
        limit=min(config.mining.hard_positive_limit, len(all_optimization_positives)),
    )
    index_by_id = {posting_id: index for index, posting_id in enumerate(dataset.posting_ids)}
    optimization_set = set(partition.optimization_indices)
    mined_negatives = sorted(
        load_hard_negative_manifest(config.source.candidate.source.mined_manifest_path),
        key=lambda pair: (-pair.pair_probability, pair.left_posting_id, pair.right_posting_id),
    )
    hard_negatives = [
        (index_by_id[pair.left_posting_id], index_by_id[pair.right_posting_id])
        for pair in mined_negatives
        if pair.left_posting_id in index_by_id
        and pair.right_posting_id in index_by_id
        and index_by_id[pair.left_posting_id] in optimization_set
        and index_by_id[pair.right_posting_id] in optimization_set
    ][: config.mining.hard_negative_limit]
    provider = MixedPairBatchProvider(
        hard_positives=hard_positives,
        hard_negatives=hard_negatives,
        random_positives=positive_pairs(dataset.labels, partition.optimization_indices),
        allowed_indices=partition.optimization_indices,
        labels=dataset.labels,
        seed=config.seed,
    )
    holdout_pairs, holdout_targets = _holdout_candidate_pairs(
        joint,
        dataset.labels,
        partition.holdout_indices,
        candidate_k=config.evaluation.holdout_candidate_k,
    )
    baseline_holdout_probabilities = _score_index_pairs(
        model,
        joint,
        holdout_pairs,
        batch_size=config.mining.scoring_batch_size,
        device=device,
    )
    baseline_holdout = _pair_metrics(
        holdout_targets,
        baseline_holdout_probabilities,
        minimum_precision=config.evaluation.minimum_holdout_precision,
    )
    LOGGER.info("Hard-positive stage 3/5: evaluating frozen validation graph")
    baseline_validation = _validation_graph(model, config, device)

    model.fusion.requires_grad_(False)
    model.fusion.eval()
    optimizer = AdamW(
        model.pair_head.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    best_epoch = -1
    best_holdout = baseline_holdout
    history: list[dict[str, float]] = []
    epochs_without_improvement = 0
    LOGGER.info(
        "Hard-positive stage 4/5: fine-tuning pair head for at most %d epochs",
        config.training.epochs,
    )
    for epoch in range(config.training.epochs):
        model.pair_head.train()
        total_loss = 0.0
        for batch_index in range(config.training.batches_per_epoch):
            left, right, targets = provider.sample(
                epoch=epoch,
                batch_index=batch_index,
                batch_size=config.training.batch_size,
                config=config.training,
            )
            optimizer.zero_grad(set_to_none=True)
            logits = model.pair_logits(joint[left].to(device), joint[right].to(device))
            loss = F.binary_cross_entropy_with_logits(logits, targets.to(device))
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite hard-positive pair loss")
            loss.backward()  # type: ignore[no-untyped-call]
            torch.nn.utils.clip_grad_norm_(
                model.pair_head.parameters(), config.training.gradient_clip_norm
            )
            optimizer.step()
            total_loss += float(loss.detach().cpu())
        holdout_probabilities = _score_index_pairs(
            model,
            joint,
            holdout_pairs,
            batch_size=config.mining.scoring_batch_size,
            device=device,
        )
        holdout = _pair_metrics(
            holdout_targets,
            holdout_probabilities,
            minimum_precision=config.evaluation.minimum_holdout_precision,
        )
        record = {
            "epoch": float(epoch),
            "train_loss": total_loss / config.training.batches_per_epoch,
            **{f"holdout_{key}": value for key, value in holdout.items()},
        }
        history.append(record)
        LOGGER.info(
            "epoch %d/%d loss=%.5f holdout_precision=%.5f recall=%.5f f1=%.5f",
            epoch + 1,
            config.training.epochs,
            record["train_loss"],
            holdout["precision"],
            holdout["recall"],
            holdout["f1"],
        )
        if holdout["f1"] > best_holdout["f1"]:
            best_epoch = epoch
            best_holdout = holdout
            best_state = {
                name: value.detach().cpu().clone() for name, value in model.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if epochs_without_improvement >= config.training.early_stopping_patience:
            break
    model.load_state_dict(best_state)
    model.to(device).eval()
    LOGGER.info("Hard-positive stage 5/5: evaluating selected checkpoint on validation")
    selected_validation = _validation_graph(model, config, device)
    baseline_cluster = baseline_validation["selected"]["clustering"]
    selected_cluster = selected_validation["selected"]["clustering"]
    f1_delta = (
        selected_cluster["pairwise"]["f1"] - baseline_cluster["pairwise"]["f1"]
    )
    checks = {
        "holdout_checkpoint_improved": best_epoch >= 0,
        "validation_precision": (
            selected_cluster["pairwise"]["precision"]
            >= config.evaluation.minimum_validation_precision
        ),
        "validation_false_merge": (
            selected_cluster["false_merge_pair_rate"]
            <= config.evaluation.maximum_validation_false_merge_rate
        ),
        "validation_f1_delta": f1_delta >= config.evaluation.minimum_validation_f1_delta,
    }
    status = "accepted_validation_only" if all(checks.values()) else "not_accepted_validation_only"
    commit, dirty = _git_state()
    run: dict[str, Any] = {
        "pipeline_version": "pair_head.hard_positive_finetuning.v1",
        "status": status,
        "provenance": {
            "config_sha256": canonical_text_sha256(config.config_path),
            "source_checkpoint_sha256": config.source.candidate.source.checkpoint_sha256,
            "hard_negative_manifest_sha256": sha256_file(
                config.source.candidate.source.mined_manifest_path
            ),
            "git_commit": commit,
            "git_dirty": dirty,
            "seed": config.seed,
            "device": str(device),
        },
        "data": {
            "source_split": "train",
            "optimization_listings": len(partition.optimization_indices),
            "holdout_listings": len(partition.holdout_indices),
            "optimization_groups": partition.optimization_groups,
            "holdout_groups": partition.holdout_groups,
            "label_group_overlap": 0,
            "test_accessed": False,
        },
        "mining": {
            "candidate_positive_pairs": len(all_optimization_positives),
            "selected_hard_positives": len(hard_positives),
            "selected_hard_negatives": len(hard_negatives),
            "hard_positive_probability": {
                "minimum": float(np.min(positive_probabilities)),
                "median": float(np.median(positive_probabilities)),
                "maximum_selected": float(
                    np.partition(positive_probabilities, len(hard_positives) - 1)[
                        len(hard_positives) - 1
                    ]
                ),
            },
        },
        "training": {
            "pair_head_only": True,
            "encoders_frozen": True,
            "fusion_frozen": True,
            "best_epoch": best_epoch,
            "history": history,
        },
        "holdout": {"baseline": baseline_holdout, "selected_checkpoint": best_holdout},
        "validation_graph": {
            "baseline": baseline_validation["selected"],
            "selected_checkpoint": selected_validation["selected"],
            "pairwise_f1_delta": f1_delta,
        },
        "acceptance": checks,
        "test": {"status": "disabled"},
        "runtime_seconds": time.perf_counter() - started,
    }
    _save_checkpoint_atomic(
        config.artifacts.checkpoint,
        {
            "checkpoint_version": "pair_head.hard_positive_finetuning.v1",
            "model_state": best_state,
            "model_spec": asdict(
                config.source.candidate.source.experiment.source.experiment.model_spec
            ),
            "best_epoch": best_epoch,
            "holdout": best_holdout,
            "config_sha256": canonical_text_sha256(config.config_path),
            "source_checkpoint_sha256": config.source.candidate.source.checkpoint_sha256,
        },
    )
    _write_text_atomic(config.artifacts.metrics, json.dumps(run, indent=2, sort_keys=True) + "\n")
    _write_text_atomic(config.artifacts.report, _render_report(run))
    return {
        "status": status,
        "best_epoch": best_epoch,
        "holdout_f1": best_holdout["f1"],
        "validation_pairwise_precision": selected_cluster["pairwise"]["precision"],
        "validation_pairwise_recall": selected_cluster["pairwise"]["recall"],
        "validation_pairwise_f1": selected_cluster["pairwise"]["f1"],
        "validation_false_merge_pair_rate": selected_cluster["false_merge_pair_rate"],
        "test_accessed": False,
        "metrics": str(config.artifacts.metrics),
    }
