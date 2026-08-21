"""Frozen-encoder extraction cache and lightweight Phase 5 training dataset."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from shopee_match.errors import ConfigurationError, DataValidationError
from shopee_match.evaluation.protocol import load_splits
from shopee_match.models import (
    ImageEncoderSpec,
    ScratchResidualImageEncoder,
    ScratchTextCNN,
    TextEncoderSpec,
)
from shopee_match.training.image_data import ImagePreprocessor, ProductImageDataset
from shopee_match.training.multimodal_config import (
    MultimodalExperimentConfig,
    load_multimodal_experiment_config,
)
from shopee_match.training.text_data import CharacterVocabulary, ProductTextDataset

FloatArray = NDArray[np.float32]
LOGGER = logging.getLogger(__name__)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def _load_frozen_encoders(
    config: MultimodalExperimentConfig, device: torch.device
) -> tuple[ScratchResidualImageEncoder, ScratchTextCNN, CharacterVocabulary, int]:
    image_payload = torch.load(
        config.frozen.image_config.checkpoint.path, map_location="cpu", weights_only=False
    )
    text_payload = torch.load(
        config.frozen.text_config.checkpoint.path, map_location="cpu", weights_only=False
    )
    if image_payload.get("checkpoint_version") != "phase3.scratch_image_checkpoint.v1":
        raise ConfigurationError("Unsupported frozen image checkpoint")
    if text_payload.get("checkpoint_version") != "phase4.scratch_text_checkpoint.v1":
        raise ConfigurationError("Unsupported frozen text checkpoint")
    if image_payload.get("split_manifest_sha256") != text_payload.get("split_manifest_sha256"):
        raise ConfigurationError("Frozen encoders were not trained from the same split manifest")

    image_spec = ImageEncoderSpec(**image_payload["model_spec"])
    text_spec = TextEncoderSpec(**text_payload["model_spec"])
    vocabulary = CharacterVocabulary.from_dict(cast(dict[str, object], text_payload["vocabulary"]))
    image_model = ScratchResidualImageEncoder(image_spec)
    text_model = ScratchTextCNN(
        len(vocabulary.tokens), text_spec, padding_index=vocabulary.padding_index
    )
    image_model.load_state_dict(image_payload["model_state"])
    text_model.load_state_dict(text_payload["model_state"])
    image_model.requires_grad_(False).eval().to(device)
    text_model.requires_grad_(False).eval().to(device)
    return image_model, text_model, vocabulary, int(text_payload["maximum_length"])


def _extract_image_embeddings(
    model: ScratchResidualImageEncoder,
    loader: DataLoader[dict[str, Tensor | str]],
    device: torch.device,
    stage: str,
) -> tuple[tuple[str, ...], FloatArray, float]:
    identifiers: list[str] = []
    arrays: list[FloatArray] = []
    started = time.perf_counter()
    with torch.inference_mode():
        milestones = {max(1, round(len(loader) * step / 4)) for step in range(1, 5)}
        for batch_index, raw_batch in enumerate(loader, start=1):
            batch = cast(Mapping[str, Any], raw_batch)
            arrays.append(
                cast(FloatArray, model(_tensor_batch(batch, "image", device)).cpu().numpy())
            )
            identifiers.extend(_posting_ids(batch))
            if batch_index in milestones:
                LOGGER.info(
                    "%s image extraction: %d/%d batches (%d%%)",
                    stage,
                    batch_index,
                    len(loader),
                    round(100 * batch_index / len(loader)),
                )
    if not arrays:
        raise DataValidationError("Cannot cache an empty image split")
    return tuple(identifiers), np.concatenate(arrays), time.perf_counter() - started


def _extract_text_embeddings(
    model: ScratchTextCNN,
    loader: DataLoader[dict[str, Tensor | str]],
    device: torch.device,
    stage: str,
) -> tuple[tuple[str, ...], FloatArray, float]:
    identifiers: list[str] = []
    arrays: list[FloatArray] = []
    started = time.perf_counter()
    with torch.inference_mode():
        milestones = {max(1, round(len(loader) * step / 4)) for step in range(1, 5)}
        for batch_index, raw_batch in enumerate(loader, start=1):
            batch = cast(Mapping[str, Any], raw_batch)
            arrays.append(
                cast(
                    FloatArray,
                    model(
                        _tensor_batch(batch, "token_ids", device),
                        _tensor_batch(batch, "length", device),
                    )
                    .cpu()
                    .numpy(),
                )
            )
            identifiers.extend(_posting_ids(batch))
            if batch_index in milestones:
                LOGGER.info(
                    "%s text extraction: %d/%d batches (%d%%)",
                    stage,
                    batch_index,
                    len(loader),
                    round(100 * batch_index / len(loader)),
                )
    if not arrays:
        raise DataValidationError("Cannot cache an empty text split")
    return tuple(identifiers), np.concatenate(arrays), time.perf_counter() - started


def _cache_contract(config: MultimodalExperimentConfig, split_name: str) -> dict[str, object]:
    return {
        "version": "phase5.frozen_multimodal_cache.v1",
        "split": split_name,
        "metadata_sha256": _sha256(config.data.metadata_csv),
        "manifest_sha256": _sha256(config.data.split_manifest),
        "image_checkpoint_sha256": config.frozen.image_config.checkpoint.sha256,
        "text_checkpoint_sha256": config.frozen.text_config.checkpoint.sha256,
        "image_size": config.frozen.image_config.training_experiment.image_size,
        "maximum_length": config.frozen.text_config.training_experiment.tokenization.maximum_length,
        "image_normalization": "fixed_half_range",
        "title_normalization": "nfkc_casefold_identity_preserving",
    }


def _write_cache_atomic(
    path: Path,
    *,
    posting_ids: tuple[str, ...],
    labels: tuple[str, ...],
    image_embeddings: FloatArray,
    text_embeddings: FloatArray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            posting_ids=np.asarray(posting_ids, dtype=str),
            labels=np.asarray(labels, dtype=str),
            image_embeddings=image_embeddings.astype(np.float32, copy=False),
            text_embeddings=text_embeddings.astype(np.float32, copy=False),
        )
    temporary.replace(path)


def _write_json_atomic(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _validate_cache(
    cache_path: Path, metadata_path: Path, expected_contract: dict[str, object]
) -> dict[str, object]:
    try:
        metadata = cast(dict[str, object], json.loads(metadata_path.read_text(encoding="utf-8")))
        with np.load(cache_path, allow_pickle=False) as payload:
            posting_ids = payload["posting_ids"]
            labels = payload["labels"]
            image = payload["image_embeddings"]
            text = payload["text_embeddings"]
    except (OSError, KeyError, ValueError, json.JSONDecodeError) as exc:
        raise DataValidationError(f"Cannot read frozen multimodal cache {cache_path}") from exc
    if metadata.get("contract") != expected_contract:
        raise DataValidationError(f"Frozen multimodal cache contract changed: {cache_path}")
    if (
        image.ndim != 2
        or text.ndim != 2
        or image.shape[0] != text.shape[0]
        or posting_ids.shape != labels.shape
        or posting_ids.shape[0] != image.shape[0]
        or len(set(posting_ids.tolist())) != posting_ids.shape[0]
    ):
        raise DataValidationError(f"Frozen multimodal cache arrays are inconsistent: {cache_path}")
    if not np.isfinite(image).all() or not np.isfinite(text).all():
        raise DataValidationError(
            f"Frozen multimodal cache contains non-finite values: {cache_path}"
        )
    return metadata


def prepare_frozen_multimodal_cache(config_path: Path) -> dict[str, object]:
    """Extract train/validation embeddings once; never access the held-out test split."""
    config = load_multimodal_experiment_config(config_path)
    device = _resolve_device(config.training.device)
    splits = load_splits(config.data.metadata_csv, config.data.split_manifest)
    image_model, text_model, vocabulary, maximum_length = _load_frozen_encoders(config, device)
    LOGGER.info("frozen encoder cache: device=%s test=disabled", device)
    results: dict[str, object] = {}
    for split_name in ("train", "validation"):
        split = splits[split_name]
        cache_path = config.cache.root / f"{split_name}.npz"
        metadata_path = config.cache.root / f"{split_name}.json"
        contract = _cache_contract(config, split_name)
        if cache_path.exists() or metadata_path.exists():
            if not cache_path.exists() or not metadata_path.exists():
                raise DataValidationError(f"Incomplete multimodal cache for {split_name}")
            metadata = _validate_cache(cache_path, metadata_path, contract)
            results[split_name] = {"status": "reused", **metadata}
            LOGGER.info("%s cache reused: %d listings", split_name, metadata["listings"])
            continue

        image_dataset = ProductImageDataset.for_split(
            split,
            config.data.image_dir,
            ImagePreprocessor(
                config.frozen.image_config.training_experiment.image_size,
                training=False,
                seed=config.seed,
            ),
        )
        text_dataset = ProductTextDataset(split, vocabulary, maximum_length)
        image_loader = DataLoader(
            image_dataset,
            batch_size=config.cache.batch_size,
            shuffle=False,
            num_workers=config.cache.num_workers,
            pin_memory=device.type == "cuda",
        )
        text_loader = DataLoader(
            text_dataset,
            batch_size=config.cache.batch_size,
            shuffle=False,
            num_workers=config.cache.num_workers,
            pin_memory=device.type == "cuda",
        )
        LOGGER.info("%s cache: extracting %d listings", split_name, len(split.items))
        image_ids, image_embeddings, image_seconds = _extract_image_embeddings(
            image_model, image_loader, device, split_name
        )
        text_ids, text_embeddings, text_seconds = _extract_text_embeddings(
            text_model, text_loader, device, split_name
        )
        expected_ids = tuple(item.posting_id for item in split.items)
        if image_ids != text_ids or image_ids != expected_ids:
            raise DataValidationError("Image/text cache extraction order differs from the split")
        labels = tuple(split.label_by_id[posting_id] for posting_id in image_ids)
        _write_cache_atomic(
            cache_path,
            posting_ids=image_ids,
            labels=labels,
            image_embeddings=image_embeddings,
            text_embeddings=text_embeddings,
        )
        metadata = {
            "contract": contract,
            "listings": len(image_ids),
            "image_embedding_dim": image_embeddings.shape[1],
            "text_embedding_dim": text_embeddings.shape[1],
            "image_extraction_seconds": image_seconds,
            "text_extraction_seconds": text_seconds,
            "cache_bytes": cache_path.stat().st_size,
        }
        _write_json_atomic(metadata_path, metadata)
        results[split_name] = {"status": "created", **metadata}
        LOGGER.info(
            "%s cache complete: image=%ds text=%ds path=%s",
            split_name,
            round(image_seconds),
            round(text_seconds),
            cache_path,
        )
    return {"status": "complete", "device": str(device), "test_accessed": False, "splits": results}


class CachedMultimodalDataset(Dataset[dict[str, Tensor | str]]):
    """In-memory frozen image/text embeddings aligned by posting ID."""

    def __init__(self, cache_path: Path) -> None:
        try:
            with np.load(cache_path, allow_pickle=False) as payload:
                self.posting_ids = tuple(str(value) for value in payload["posting_ids"].tolist())
                self.labels = tuple(str(value) for value in payload["labels"].tolist())
                self.image_embeddings = torch.from_numpy(
                    payload["image_embeddings"].astype(np.float32, copy=True)
                )
                self.text_embeddings = torch.from_numpy(
                    payload["text_embeddings"].astype(np.float32, copy=True)
                )
        except (OSError, KeyError, ValueError) as exc:
            raise DataValidationError(f"Cannot load multimodal cache: {cache_path}") from exc
        label_to_index = {label: index for index, label in enumerate(sorted(set(self.labels)))}
        self.label_indices = tuple(label_to_index[label] for label in self.labels)
        if len(self.posting_ids) != len(self.labels) or len(self.labels) != len(
            self.image_embeddings
        ):
            raise DataValidationError("Cached multimodal arrays have inconsistent lengths")

    def __len__(self) -> int:
        return len(self.posting_ids)

    def __getitem__(self, index: int) -> dict[str, Tensor | str]:
        return {
            "image_embedding": self.image_embeddings[index],
            "text_embedding": self.text_embeddings[index],
            "label": torch.tensor(self.label_indices[index], dtype=torch.long),
            "posting_id": self.posting_ids[index],
        }


def load_cached_multimodal_split(
    config: MultimodalExperimentConfig, split_name: str
) -> CachedMultimodalDataset:
    if split_name not in {"train", "validation"}:
        raise ConfigurationError("Phase 5 training cache exposes only train and validation")
    cache_path = config.cache.root / f"{split_name}.npz"
    metadata_path = config.cache.root / f"{split_name}.json"
    if not cache_path.exists() or not metadata_path.exists():
        raise DataValidationError(
            f"Missing {split_name} frozen cache; run the multimodal prepare command first"
        )
    _validate_cache(cache_path, metadata_path, _cache_contract(config, split_name))
    return CachedMultimodalDataset(cache_path)
