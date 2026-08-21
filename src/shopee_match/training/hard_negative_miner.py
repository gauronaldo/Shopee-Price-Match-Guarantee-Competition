"""Train-only exact retrieval and deterministic hard-negative manifest generation."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader

from shopee_match.errors import ConfigurationError, DataValidationError, OutputConflictError
from shopee_match.evaluation.protocol import load_splits
from shopee_match.models import LearnedMultimodalFusion
from shopee_match.reproducibility import seed_everything
from shopee_match.training.hard_negative_config import (
    HardNegativeExperimentConfig,
    load_hard_negative_experiment_config,
)
from shopee_match.training.hard_negative_data import (
    MinedHardNegative,
    MiningCandidate,
    MiningSelectionStats,
    cap_variant_share,
    deduplicate_hard_negatives,
    hard_negative_jsonl,
    select_query_hard_negatives,
)
from shopee_match.training.multimodal_data import load_cached_multimodal_split
from shopee_match.training.multimodal_trainer import (
    _git_state,
    _resolve_device,
    extract_joint_embeddings,
)
from shopee_match.training.text_evaluation_config import sha256_file

LOGGER = logging.getLogger(__name__)


def _write_text_atomic(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def _canonical_text_sha256(path: Path) -> str:
    raw = path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(raw).hexdigest()


def exact_topk_cosine_block(
    embeddings: Tensor,
    start: int,
    stop: int,
    candidate_k: int,
) -> tuple[Tensor, Tensor]:
    """Return exact non-self cosine Top-K for a contiguous query block."""
    if embeddings.ndim != 2 or start < 0 or stop > len(embeddings) or start >= stop:
        raise ValueError("invalid exact-retrieval block")
    if candidate_k <= 0 or candidate_k >= len(embeddings):
        raise ValueError("candidate_k must be inside [1, listings - 1]")
    normalized = torch.nn.functional.normalize(embeddings, p=2, dim=1)
    scores = normalized[start:stop] @ normalized.T
    local_rows = torch.arange(stop - start, device=embeddings.device)
    global_rows = torch.arange(start, stop, device=embeddings.device)
    scores[local_rows, global_rows] = -torch.inf
    values, indices = torch.topk(scores, k=candidate_k, dim=1, largest=True, sorted=True)
    return indices, values


def load_phase5_source_model(
    config: HardNegativeExperimentConfig, device: torch.device
) -> LearnedMultimodalFusion:
    checkpoint = torch.load(config.source.checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("checkpoint_version") != "phase5.scratch_multimodal_checkpoint.v1":
        raise ConfigurationError("Unsupported Phase 5 source checkpoint")
    if checkpoint.get("model_spec") != asdict(config.source.experiment.model_spec):
        raise ConfigurationError("Phase 5 checkpoint architecture differs from its frozen config")
    selection = config.source.metrics["selection"]
    if checkpoint.get("best_epoch") != selection["best_epoch"] or not np.isclose(
        float(checkpoint.get("best_metric", float("nan"))),
        float(selection["best_metric"]),
        rtol=0.0,
        atol=1e-12,
    ):
        raise ConfigurationError("Phase 5 checkpoint and metrics disagree on model selection")
    model = LearnedMultimodalFusion(config.source.experiment.model_spec)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model.to(device)


def _score_pairs(
    model: LearnedMultimodalFusion,
    joint: Tensor,
    query_indices: list[int],
    candidate_indices: list[int],
    *,
    batch_size: int,
) -> list[float]:
    probabilities: list[float] = []
    with torch.inference_mode():
        for start in range(0, len(query_indices), batch_size):
            stop = start + batch_size
            left = joint[query_indices[start:stop]]
            right = joint[candidate_indices[start:stop]]
            probabilities.extend(torch.sigmoid(model.pair_logits(left, right)).cpu().tolist())
    return probabilities


def _render_mining_report(metadata: dict[str, Any]) -> str:
    counts = metadata["selection_counts"]
    score = metadata["selected_pair_probability"]
    return f"""# Hard-negative mining pilot

## Contract

The frozen Phase 5 model retrieved neighbours from the **train split only**. Labels were used only
after retrieval to retain different-product pairs. Test data was not loaded. Cross-label pairs with
the same pHash or exactly normalized title were excluded as possible false negatives or fragmented
labels. Symmetric duplicates were collapsed into one canonical pair.

## Mining result

| Measure | Value |
|---|---:|
| Train listings queried | {metadata["train_listings"]:,} |
| Exact neighbours per query | {metadata["candidate_k"]} |
| Raw retrieved candidates | {counts["candidates_seen"]:,} |
| Eligible cross-label candidates | {counts["eligible"]:,} |
| Final unique mined pairs | {metadata["mined_pairs"]:,} |
| Variant-conflict pairs | {metadata["variant_conflict_pairs"]:,} |
| Symmetric duplicates removed | {counts["symmetric_duplicates_removed"]:,} |
| Pair probability median / P95 | {score["median"]:.5f} / {score["p95"]:.5f} |
| Mining wall time | {metadata["wall_time_seconds"]:.2f} s |

## False-negative guards

| Exclusion | Count |
|---|---:|
| Same label (true train positive) | {counts["excluded_same_label"]:,} |
| Outside probability bounds | {counts["excluded_probability"]:,} |
| Same pHash across labels | {counts["excluded_same_phash"]:,} |
| Exact normalized title across labels | {counts["excluded_exact_title"]:,} |
| Variant candidates removed by final share cap | {counts["variant_quota_removed"]:,} |

## Provenance

- Source checkpoint SHA-256: `{metadata["source"]["checkpoint_sha256"]}`
- Source config SHA-256 (canonical LF): `{metadata["source"]["config_sha256"]}`
- Mined manifest SHA-256: `{metadata["manifest_sha256"]}`
- Split: `train`
- Test accessed: `false`

The section above records mining evidence. The Phase 6 training command appends its validation
comparison below without changing the mined manifest.
"""


def mine_hard_negatives(config_path: Path) -> dict[str, object]:
    """Mine a versioned Phase 6 pair manifest without loading validation or test."""
    config = load_hard_negative_experiment_config(config_path)
    existing = [
        str(path)
        for path in (config.artifacts.manifest, config.artifacts.manifest_metadata)
        if path.exists()
    ]
    if existing:
        raise OutputConflictError(
            "Refusing to overwrite mined Phase 6 evidence: " + ", ".join(existing)
        )
    seed_everything(config.seed, deterministic=config.training.deterministic)
    device = _resolve_device(config.training.device)
    source = config.source.experiment
    train_split = load_splits(source.data.metadata_csv, source.data.split_manifest)["train"]
    train_dataset = load_cached_multimodal_split(source, "train")
    expected_ids = tuple(item.posting_id for item in train_split.items)
    if train_dataset.posting_ids != expected_ids:
        raise DataValidationError("Train cache order differs from the frozen train split")
    loader = DataLoader(train_dataset, batch_size=512, shuffle=False, num_workers=0)
    model = load_phase5_source_model(config, device)
    started = time.perf_counter()
    posting_ids, _image, _text, joint_array, extraction_seconds = extract_joint_embeddings(
        model, loader, device
    )
    if posting_ids != expected_ids:
        raise DataValidationError("Extracted train embeddings do not align with train IDs")
    joint = torch.from_numpy(joint_array).to(device)
    selected: list[MinedHardNegative] = []
    totals = MiningSelectionStats()
    block_size = config.mining.similarity_block_size
    block_count = (len(joint) + block_size - 1) // block_size
    LOGGER.info(
        "Phase 6 mining: device=%s train=%d exact_k=%d blocks=%d validation/test=disabled",
        device,
        len(joint),
        config.mining.candidate_k,
        block_count,
    )
    for block_number, start in enumerate(range(0, len(joint), block_size), start=1):
        stop = min(start + block_size, len(joint))
        indices, cosine = exact_topk_cosine_block(joint, start, stop, config.mining.candidate_k)
        query_indices = [
            query_index
            for query_index in range(start, stop)
            for _ in range(config.mining.candidate_k)
        ]
        candidate_indices = indices.flatten().cpu().tolist()
        probabilities = _score_pairs(
            model,
            joint,
            query_indices,
            candidate_indices,
            batch_size=config.mining.pair_batch_size,
        )
        cosine_values = cosine.flatten().cpu().tolist()
        for local_index, query_index in enumerate(range(start, stop)):
            offset = local_index * config.mining.candidate_k
            rows = [
                MiningCandidate(
                    query_index=query_index,
                    candidate_index=int(candidate_indices[offset + rank]),
                    cosine_similarity=float(cosine_values[offset + rank]),
                    pair_probability=float(probabilities[offset + rank]),
                )
                for rank in range(config.mining.candidate_k)
            ]
            query_selected, query_stats = select_query_hard_negatives(
                query_index,
                rows,
                train_split.items,
                train_split.label_by_id,
                negatives_per_query=config.mining.negatives_per_query,
                minimum_pair_probability=config.mining.minimum_pair_probability,
                maximum_pair_probability=config.mining.maximum_pair_probability,
                exclude_same_phash=config.mining.exclude_same_phash,
                exclude_exact_normalized_title=config.mining.exclude_exact_normalized_title,
                variant_priority_fraction=config.mining.variant_priority_fraction,
            )
            selected.extend(query_selected)
            totals.add(query_stats)
        if block_number == 1 or block_number == block_count or block_number % 10 == 0:
            LOGGER.info(
                "mining block %d/%d: queries=%d/%d selected_before_dedup=%d",
                block_number,
                block_count,
                stop,
                len(joint),
                len(selected),
            )
    pairs, duplicates = deduplicate_hard_negatives(selected)
    totals.symmetric_duplicates_removed = duplicates
    pairs, quota_removed = cap_variant_share(pairs, config.mining.variant_priority_fraction)
    totals.variant_quota_removed = quota_removed
    if not pairs:
        raise DataValidationError("Mining produced no eligible hard-negative pair")
    manifest_content = hard_negative_jsonl(pairs)
    _write_text_atomic(config.artifacts.manifest, manifest_content)
    manifest_sha = sha256_file(config.artifacts.manifest)
    probability_values = np.asarray([pair.pair_probability for pair in pairs], dtype=np.float64)
    commit, dirty = _git_state()
    metadata: dict[str, Any] = {
        "pipeline_version": "phase6.hard_negative_mining.v1",
        "seed": config.seed,
        "split": "train",
        "test_accessed": False,
        "train_listings": len(train_dataset),
        "candidate_k": config.mining.candidate_k,
        "mined_pairs": len(pairs),
        "variant_conflict_pairs": sum(pair.variant_conflict for pair in pairs),
        "selection_counts": asdict(totals),
        "selected_pair_probability": {
            "minimum": float(probability_values.min()),
            "median": float(np.median(probability_values)),
            "p95": float(np.quantile(probability_values, 0.95)),
            "maximum": float(probability_values.max()),
        },
        "source": {
            "config_sha256": config.source.multimodal_config_sha256,
            "checkpoint_sha256": config.source.checkpoint_sha256,
            "metrics_sha256": config.source.metrics_sha256,
            "split_manifest_sha256": sha256_file(source.data.split_manifest),
        },
        "phase6_config_sha256": _canonical_text_sha256(config.config_path),
        "manifest_sha256": manifest_sha,
        "embedding_extraction_seconds": extraction_seconds,
        "wall_time_seconds": time.perf_counter() - started,
        "git_commit": commit,
        "git_dirty": dirty,
        "device": str(device),
    }
    _write_text_atomic(
        config.artifacts.manifest_metadata,
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
    )
    _write_text_atomic(config.artifacts.report, _render_mining_report(metadata))
    return {
        "status": "complete",
        "split": "train",
        "manifest": str(config.artifacts.manifest),
        "manifest_sha256": manifest_sha,
        "mined_pairs": len(pairs),
        "report": str(config.artifacts.report),
        "test_accessed": False,
    }
