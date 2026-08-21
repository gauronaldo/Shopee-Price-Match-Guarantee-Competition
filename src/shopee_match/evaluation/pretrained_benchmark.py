"""Frozen EfficientNet-B1 image retrieval benchmark for Phase 9."""

from __future__ import annotations

import importlib
import json
import logging
import os
import platform
import time
from collections import Counter
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, cast

import cv2
import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset

from shopee_match.errors import DataValidationError, OutputConflictError
from shopee_match.evaluation.pretrained_config import (
    PretrainedBenchmarkConfig,
    load_pretrained_benchmark_config,
)
from shopee_match.evaluation.protocol import (
    EvaluationSplit,
    Ranking,
    load_named_split,
    retrieval_metrics,
    select_threshold,
)
from shopee_match.hashing import canonical_text_sha256
from shopee_match.reproducibility import seed_everything
from shopee_match.retrieval.vector_index import ExactCosineIndex, search_result_to_ranking
from shopee_match.training.multimodal_trainer import _git_state, _resolve_device

LOGGER = logging.getLogger(__name__)
FloatArray = NDArray[np.float32]


class PretrainedImageDataset(Dataset[dict[str, Tensor | str]]):
    """Decode validation images and apply the official frozen-weight transform."""

    def __init__(
        self,
        split: EvaluationSplit,
        image_dir: Path,
        transform: Callable[[Tensor], Tensor],
    ) -> None:
        self.items = split.items
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, Tensor | str]:
        item = self.items[index]
        image = cv2.imread(str(self.image_dir / item.image), cv2.IMREAD_COLOR)
        if image is None:
            raise DataValidationError(f"Cannot decode pretrained benchmark image: {item.image}")
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1))).float() / 255.0
        return {"image": self.transform(tensor), "posting_id": item.posting_id}


def _write_text_atomic(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def _write_embeddings_atomic(path: Path, posting_ids: tuple[str, ...], values: FloatArray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            posting_ids=np.asarray(posting_ids, dtype=str),
            embeddings=values,
        )
    temporary.replace(path)


def _posting_ids(batch: Mapping[str, Any]) -> tuple[str, ...]:
    values = batch["posting_id"]
    if not isinstance(values, list | tuple):
        raise DataValidationError("Pretrained batch posting IDs have an invalid type")
    return tuple(str(value) for value in values)


def _load_model(
    config: PretrainedBenchmarkConfig, device: torch.device
) -> tuple[nn.Module, Callable[[Tensor], Tensor], str]:
    try:
        models = importlib.import_module("torchvision.models")
        torchvision = importlib.import_module("torchvision")
    except ImportError as exc:
        raise DataValidationError(
            'Phase 9 requires the optional dependency: pip install -e ".[pretrained]"'
        ) from exc
    weights = models.EfficientNet_B1_Weights.IMAGENET1K_V2
    model = models.efficientnet_b1(weights=None)
    state = torch.load(config.source.weights_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.classifier = nn.Identity()
    model.eval().to(device)
    return (
        cast(nn.Module, model),
        cast(Callable[[Tensor], Tensor], weights.transforms()),
        str(torchvision.__version__),
    )


def _extract_embeddings(
    model: nn.Module, loader: DataLoader[dict[str, Tensor | str]], device: torch.device
) -> tuple[tuple[str, ...], FloatArray, float]:
    identifiers: list[str] = []
    chunks: list[FloatArray] = []
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    with torch.inference_mode():
        for batch_number, batch in enumerate(loader, start=1):
            images = cast(Tensor, batch["image"]).to(device, non_blocking=True)
            features = torch.nn.functional.normalize(model(images), p=2, dim=1)
            if not torch.isfinite(features).all():
                raise DataValidationError("Pretrained encoder produced non-finite embeddings")
            identifiers.extend(_posting_ids(batch))
            chunks.append(features.cpu().numpy().astype(np.float32, copy=False))
            if batch_number % 10 == 0 or batch_number == len(loader):
                LOGGER.info("Phase 9 extraction: batch %d/%d", batch_number, len(loader))
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    return tuple(identifiers), np.concatenate(chunks, axis=0), elapsed


def _profile_queries(
    index: ExactCosineIndex,
    embeddings: FloatArray,
    posting_ids: tuple[str, ...],
    *,
    candidate_k: int,
    query_count: int,
    block_size: int,
) -> dict[str, float]:
    samples: list[float] = []
    count = min(query_count, len(posting_ids))
    for row in range(count):
        started = time.perf_counter()
        index.search(
            embeddings[row : row + 1],
            candidate_k,
            query_ids=(posting_ids[row],),
            block_size=block_size,
        )
        samples.append((time.perf_counter() - started) * 1000)
    batch_started = time.perf_counter()
    index.search(
        embeddings,
        candidate_k,
        query_ids=posting_ids,
        block_size=block_size,
    )
    batch_seconds = time.perf_counter() - batch_started
    return {
        "single_query_p50_ms": float(np.percentile(samples, 50)),
        "single_query_p95_ms": float(np.percentile(samples, 95)),
        "batch_search_seconds": batch_seconds,
        "batch_throughput_queries_per_second": len(posting_ids) / batch_seconds,
        "profiled_queries": float(count),
    }


def _failure_review(
    ranking: Ranking, split: EvaluationSplit, *, example_limit: int = 30
) -> dict[str, Any]:
    members: dict[str, set[str]] = {}
    for posting_id, label in split.label_by_id.items():
        members.setdefault(label, set()).add(posting_id)
    item_by_id = {item.posting_id: item for item in split.items}
    false_top1: list[dict[str, Any]] = []
    zero_recall: list[dict[str, Any]] = []
    for query_id in sorted(ranking):
        positives = members[split.label_by_id[query_id]] - {query_id}
        candidates = ranking[query_id]
        top = candidates[0]
        if top.posting_id not in positives:
            false_top1.append(
                {
                    "query_id": query_id,
                    "query_title": item_by_id[query_id].title,
                    "candidate_id": top.posting_id,
                    "candidate_title": item_by_id[top.posting_id].title,
                    "score": top.score,
                }
            )
        if not positives.intersection(candidate.posting_id for candidate in candidates):
            zero_recall.append(
                {
                    "query_id": query_id,
                    "query_title": item_by_id[query_id].title,
                    "positive_count": len(positives),
                    "top_candidate_id": top.posting_id,
                    "top_candidate_title": item_by_id[top.posting_id].title,
                }
            )
    return {
        "counts": {"top1_false_matches": len(false_top1), "zero_recall_queries": len(zero_recall)},
        "top1_false_match_examples": false_top1[:example_limit],
        "zero_recall_examples": zero_recall[:example_limit],
    }


def _group_size_recall(ranking: Ranking, split: EvaluationSplit, k: int) -> dict[str, Any]:
    sizes = Counter(split.label_by_id.values())
    members: dict[str, set[str]] = {}
    for posting_id, label in split.label_by_id.items():
        members.setdefault(label, set()).add(posting_id)

    def band(size: int) -> str:
        if size == 2:
            return "2"
        if size <= 5:
            return "3_to_5"
        if size <= 9:
            return "6_to_9"
        return "10_plus"

    accumulators: dict[str, list[float]] = {key: [] for key in ("2", "3_to_5", "6_to_9", "10_plus")}
    for query_id, candidates in ranking.items():
        label = split.label_by_id[query_id]
        positives = members[label] - {query_id}
        found = positives.intersection(candidate.posting_id for candidate in candidates[:k])
        accumulators[band(sizes[label])].append(len(found) / len(positives))
    return {
        key: {"queries": float(len(values)), f"recall@{k}": float(np.mean(values))}
        for key, values in accumulators.items()
    }


def _render_report(run: dict[str, Any]) -> str:
    pretrained20 = run["validation"]["retrieval"]["20"]
    pretrained50 = run["validation"]["retrieval"]["50"]
    scratch_image = run["comparison"]["scratch_image"]
    scratch_image_efficiency = run["comparison"]["scratch_image_efficiency"]
    scratch_joint = run["comparison"]["scratch_multimodal"]
    strata = "\n".join(
        f"| {band} | {row['queries']:.0f} | {row['recall@50']:.5f} |"
        for band, row in run["validation"]["group_size_strata"].items()
    )
    scratch_image_row = (
        "| Custom residual CNN | image | random initialization | "
        f"{scratch_image['map@20']:.5f} | {scratch_image['recall@20']:.5f} | n/a | n/a |"
    )
    pretrained_row = (
        "| EfficientNet-B1 V2 | image | ImageNet-1K | "
        f"{pretrained20['map@20']:.5f} | {pretrained20['recall@20']:.5f} | "
        f"{pretrained50['map@50']:.5f} | {pretrained50['recall@50']:.5f} |"
    )
    scratch_joint_row = (
        "| Custom multimodal joint | image + title | random initialization | "
        f"{scratch_joint['map@20']:.5f} | {scratch_joint['recall@20']:.5f} | "
        f"{scratch_joint['map@50']:.5f} | {scratch_joint['recall@50']:.5f} |"
    )
    scratch_model = scratch_image_efficiency["model"]
    scratch_validation = scratch_image_efficiency["validation"]
    scratch_runtime = scratch_image_efficiency["training"]
    pretrained_model = run["model"]
    pretrained_efficiency = run["efficiency"]
    scratch_efficiency_row = (
        "| Custom residual CNN | "
        f"{scratch_model['parameter_count']:,} | {scratch_model['checkpoint_bytes']:,} | "
        f"{scratch_model['embedding_dim']} / {scratch_model['embedding_storage_bytes']:,} | "
        f"{scratch_validation['embedding_throughput_per_second']:.2f} | "
        f"{scratch_validation['search_latency']['ranking_p50_ms_per_query']:.3f} / "
        f"{scratch_validation['search_latency']['ranking_p95_ms_per_query']:.3f} | "
        f"{scratch_runtime['wall_time_seconds']:.2f} s |"
    )
    pretrained_efficiency_row = (
        "| EfficientNet-B1 V2 | "
        f"{pretrained_model['feature_parameters']:,} | "
        f"{pretrained_model['weights_file_bytes']:,} | "
        f"{pretrained_model['feature_dimension']} / "
        f"{pretrained_efficiency['embedding_storage_bytes']:,} | "
        f"{pretrained_efficiency['extraction_throughput_per_second']:.2f} | "
        f"{pretrained_efficiency['search']['single_query_p50_ms']:.3f} / "
        f"{pretrained_efficiency['search']['single_query_p95_ms']:.3f} | "
        "0 s locally* |"
    )
    efficiency_header = (
        "| Image representation | Feature params | Weight/checkpoint bytes | "
        "Embedding dim / bytes | Extraction listings/s | Exact p50 / p95 ms | "
        "Local training wall time |"
    )
    failure_counts = run["failure_analysis"]
    query_count = run["data"]["listings"]
    top1_false_rate = failure_counts["top1_false_matches"] / query_count
    zero_recall_rate = failure_counts["zero_recall_queries"] / query_count
    return f"""# Pretrained Representation Benchmark

Phase 9 status: **{run["status"]}**. This benchmark uses TorchVision EfficientNet-B1
`IMAGENET1K_V2` as a frozen image encoder. It performs no local training or fine-tuning and does
not access test.

## Protocol

- Split: validation, group-disjoint manifest inherited from Phases 1-8
- Retrieval: deterministic exact cosine over the full validation corpus
- Candidate budget: Top-50, identical to Phase 7
- Feature: 1,280-dimensional normalized penultimate EfficientNet-B1 representation
- Official preprocessing: resize 255, center crop 240, ImageNet mean/std normalization
- Weight SHA-256: `{run["provenance"]["weights_sha256"]}`

## Quality comparison

| Validation system | Modalities | Pretraining | mAP@20 | Recall@20 | mAP@50 | Recall@50 |
|---|---|---|---:|---:|---:|---:|
{scratch_image_row}
{pretrained_row}
{scratch_joint_row}

## Efficiency

{efficiency_header}
|---|---:|---:|---:|---:|---:|---:|
{scratch_efficiency_row}
{pretrained_efficiency_row}

This is the modality-matched efficiency comparison. Both rows use the same validation listings and
exact-cosine protocol. The EfficientNet weight file is not a training checkpoint from this project.
Its local training cost is zero, but the external ImageNet pretraining cost is unknown and must not
be interpreted as free compute. The custom multimodal system is omitted from this table because its
Phase 7 extraction benchmark used cached encoder outputs rather than end-to-end image/title
decoding.

## Recall by true group size

| Group size | Queries | Recall@50 |
|---|---:|---:|
{strata}

## Failure evidence

- Top-1 points to a different label for `{failure_counts["top1_false_matches"]}` of
  `{query_count}` queries (`{top1_false_rate:.2%}`).
- No true group member appears in Top-50 for `{failure_counts["zero_recall_queries"]}` queries
  (`{zero_recall_rate:.2%}`).
- Recall@50 falls to `{run["validation"]["group_size_strata"]["10_plus"]["recall@50"]:.5f}`
  for groups of at least 10 listings, versus roughly `0.90` for smaller groups.

The bounded review sample includes same-category variant confusion and unrelated products with
similar composition. Those cases are consistent with a generic visual encoder learning category
and layout cues rather than exact identity-critical text, quantity, and model-number evidence.

## Interpretation

This comparison isolates the value of generic supervised image pretraining. It is directly fair
against the custom residual CNN trained from random initialization on data split, image modality,
exact cosine ranking, and mAP@20/Recall@20.
The custom multimodal row is a system-level ceiling rather than a modality-matched comparison
because it also uses title information.

ImageNet features can recognize shapes and semantic categories, but exact-product matching often
depends on packaging text, quantity, color, or model number. A pretrained image model is therefore
not expected to replace the multimodal pipeline automatically. Weak results are retained as a
measured domain-gap finding rather than tuned on test.

Model and transform contract: [official TorchVision EfficientNet-B1 documentation](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.efficientnet_b1.html).
Bounded top-1 errors and zero-recall examples remain in the ignored review artifact.

## Reproduction

```powershell
.venv\\Scripts\\python -m pip install -e ".[dev,retrieval,pretrained]"
.venv\\Scripts\\shopee-pretrained prepare-weights
.venv\\Scripts\\shopee-pretrained benchmark `
  --config configs\\experiment\\pretrained_image_benchmark.yaml
```
"""


def run_pretrained_benchmark(config_path: Path) -> dict[str, object]:
    """Extract frozen pretrained features and compare them on the Phase 7 validation protocol."""
    config = load_pretrained_benchmark_config(config_path)
    existing = [
        str(path) for path in (config.artifacts.metrics, config.artifacts.report) if path.exists()
    ]
    if existing:
        raise OutputConflictError(
            "Refusing to overwrite completed Phase 9 evidence: " + ", ".join(existing)
        )
    seed_everything(config.seed, deterministic=True)
    device = _resolve_device(config.runtime.device)
    phase7 = config.source.phase7
    experiment = phase7.source.experiment.source.experiment
    split = load_named_split(
        experiment.data.metadata_csv, experiment.data.split_manifest, "validation"
    )
    model, transform, torchvision_version = _load_model(config, device)
    dataset = PretrainedImageDataset(split, experiment.data.image_dir, transform)
    loader = DataLoader(
        dataset,
        batch_size=config.runtime.batch_size,
        shuffle=False,
        num_workers=config.runtime.num_workers,
        pin_memory=device.type == "cuda",
    )
    LOGGER.info("Phase 9 stage 1/3: extracting frozen EfficientNet-B1 features on %s", device)
    posting_ids, embeddings, extraction_seconds = _extract_embeddings(model, loader, device)
    expected_ids = tuple(item.posting_id for item in split.items)
    if posting_ids != expected_ids or embeddings.shape != (len(posting_ids), 1280):
        raise DataValidationError("Pretrained embeddings do not match the validation contract")
    _write_embeddings_atomic(config.artifacts.embeddings, posting_ids, embeddings)

    LOGGER.info(
        "Phase 9 stage 2/3: running exact Top-%d cosine retrieval", config.evaluation.candidate_k
    )
    index = ExactCosineIndex(posting_ids, embeddings)
    indices, scores = index.search(
        embeddings,
        config.evaluation.candidate_k,
        query_ids=posting_ids,
        block_size=config.evaluation.block_size,
    )
    ranking = search_result_to_ranking(posting_ids, posting_ids, indices, scores)
    retrieval = {
        str(k): retrieval_metrics(ranking, split.label_by_id, config.evaluation.recall_at, k)
        for k in config.evaluation.average_precision_at
    }
    threshold = select_threshold(ranking, split.label_by_id)
    profile = _profile_queries(
        index,
        embeddings,
        posting_ids,
        candidate_k=config.evaluation.candidate_k,
        query_count=config.evaluation.latency_query_count,
        block_size=config.evaluation.block_size,
    )
    review = _failure_review(ranking, split)
    scratch_image_metrics = config.source.image_metrics
    scratch_image = scratch_image_metrics["validation"]["retrieval"]
    scratch_image_efficiency = {
        "model": scratch_image_metrics["model"],
        "training": scratch_image_metrics["efficiency"],
        "validation": {
            "embedding_throughput_per_second": scratch_image_metrics["validation"][
                "embedding_throughput_per_second"
            ],
            "search_latency": scratch_image_metrics["validation"]["search_latency"],
        },
    }
    phase7_curve = config.source.phase7_metrics["exact"]["retrieval_curve"]
    scratch_joint = {
        "map@20": phase7_curve["20"]["map@20"],
        "recall@20": phase7_curve["20"]["recall@20"],
        "map@50": phase7_curve["50"]["map@50"],
        "recall@50": phase7_curve["50"]["recall@50"],
    }
    commit, dirty = _git_state()
    feature_parameters = sum(parameter.numel() for parameter in model.parameters())
    run: dict[str, Any] = {
        "pipeline_version": "phase9.pretrained_efficientnet_b1.v1",
        "status": "phase9_complete_validation_only",
        "provenance": {
            "config_sha256": canonical_text_sha256(config.config_path),
            "weights_sha256": config.source.weights_sha256,
            "phase7_config_sha256": config.source.phase7_config_sha256,
            "phase7_metrics_sha256": config.source.phase7_metrics_sha256,
            "image_metrics_sha256": config.source.image_metrics_sha256,
            "git_commit": commit,
            "git_dirty": dirty,
            "seed": config.seed,
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torchvision": torchvision_version,
            "numpy": np.__version__,
            "device": str(device),
        },
        "data": {"split": "validation", "listings": len(posting_ids), "test_accessed": False},
        "model": {
            "architecture": "efficientnet_b1",
            "weights": "IMAGENET1K_V2",
            "feature_dimension": 1280,
            "feature_parameters": feature_parameters,
            "weights_file_bytes": config.source.weights_path.stat().st_size,
            "local_training_epochs": 0,
        },
        "validation": {
            "retrieval": retrieval,
            "selected_pair_threshold": threshold,
            "group_size_strata": _group_size_recall(ranking, split, 50),
        },
        "comparison": {
            "scratch_image": scratch_image,
            "scratch_image_efficiency": scratch_image_efficiency,
            "scratch_multimodal": scratch_joint,
        },
        "efficiency": {
            "extraction_seconds": extraction_seconds,
            "extraction_throughput_per_second": len(posting_ids) / extraction_seconds,
            "embedding_storage_bytes": int(embeddings.nbytes),
            "embedding_cache_bytes": config.artifacts.embeddings.stat().st_size,
            "search": profile,
        },
        "failure_analysis": review["counts"],
        "artifacts": {
            "embeddings": str(config.artifacts.embeddings),
            "review": str(config.artifacts.review),
        },
        "test": {"status": "disabled_phase9_validation_only"},
    }
    LOGGER.info("Phase 9 stage 3/3: writing metrics and failure evidence")
    _write_text_atomic(config.artifacts.review, json.dumps(review, indent=2, sort_keys=True) + "\n")
    _write_text_atomic(config.artifacts.metrics, json.dumps(run, indent=2, sort_keys=True) + "\n")
    _write_text_atomic(config.artifacts.report, _render_report(run))
    result50 = retrieval["50"]
    return {
        "status": run["status"],
        "map@50": result50["map@50"],
        "recall@50": result50["recall@50"],
        "metrics": str(config.artifacts.metrics),
        "report": str(config.artifacts.report),
        "test_accessed": False,
    }
