"""Validation-only failure review for a trained scratch text checkpoint."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import torch
from torch.utils.data import DataLoader

from shopee_match.errors import ConfigurationError
from shopee_match.evaluation.embedding_retrieval import rank_cosine_embeddings_profiled
from shopee_match.evaluation.protocol import load_splits
from shopee_match.evaluation.text_failure_analysis import build_text_failure_review
from shopee_match.models import ScratchTextCNN
from shopee_match.training.text_config import load_text_experiment_config
from shopee_match.training.text_data import CharacterVocabulary, ProductTextDataset
from shopee_match.training.text_trainer import (
    _resolve_device,
    _sha256,
    _write_text_atomic,
    extract_text_embeddings,
)


def _model_spec_payload(config_path: Path) -> dict[str, Any]:
    config = load_text_experiment_config(config_path)
    spec = config.model_spec
    return {
        "character_embedding_dim": spec.character_embedding_dim,
        "convolution_channels": spec.convolution_channels,
        "kernel_sizes": spec.kernel_sizes,
        "projection_hidden_dim": spec.projection_hidden_dim,
        "embedding_dim": spec.embedding_dim,
        "dropout": spec.dropout,
    }


def run_text_validation_failure_analysis(config_path: Path) -> dict[str, Any]:
    """Extract validation rankings and write a local review manifest without using test."""
    config = load_text_experiment_config(config_path)
    checkpoint_path = config.artifacts.root / "best.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("checkpoint_version") != "phase4.scratch_text_checkpoint.v1":
        raise ConfigurationError("Unsupported text checkpoint version")
    if checkpoint.get("split_manifest_sha256") != _sha256(config.data.split_manifest):
        raise ConfigurationError("Checkpoint split manifest differs from analysis data")
    if checkpoint.get("model_spec") != _model_spec_payload(config_path):
        raise ConfigurationError("Checkpoint architecture differs from the training config")
    vocabulary = CharacterVocabulary.from_dict(cast(dict[str, object], checkpoint["vocabulary"]))
    splits = load_splits(config.data.metadata_csv, config.data.split_manifest)
    validation_split = splits["validation"]
    dataset = ProductTextDataset(validation_split, vocabulary, config.tokenization.maximum_length)
    device = _resolve_device(config.training.device)
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=0)
    model = ScratchTextCNN(
        len(vocabulary.tokens), config.model_spec, padding_index=vocabulary.padding_index
    )
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    posting_ids, embeddings, _seconds = extract_text_embeddings(model, loader, device)
    ranking, _latency = rank_cosine_embeddings_profiled(
        posting_ids, embeddings, config.evaluation.candidate_k
    )
    review = build_text_failure_review(ranking, validation_split)
    output_path = config.artifacts.root / "validation_failure_review.json"
    _write_text_atomic(
        output_path, json.dumps(review, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )
    return {
        "status": "complete",
        "split": "validation",
        "output": str(output_path),
        "top1_false_match": len(review["top1_false_match"]),
        "retrieval_miss": len(review["retrieval_miss"]),
        "top1_success": len(review["top1_success"]),
        "test_accessed": False,
    }
