"""Phase 7 validation-only exact and FAISS candidate-retrieval benchmark."""

from __future__ import annotations

import importlib
import json
import logging
import os
import platform
import time
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from torch.utils.data import DataLoader

from shopee_match.errors import ConfigurationError, DataValidationError, OutputConflictError
from shopee_match.evaluation.protocol import (
    EvaluationSplit,
    Ranking,
    load_named_split,
    retrieval_metrics,
)
from shopee_match.hashing import canonical_text_sha256, sha256_file
from shopee_match.models import LearnedMultimodalFusion
from shopee_match.reproducibility import seed_everything
from shopee_match.retrieval.config import (
    CandidateRetrievalConfig,
    load_candidate_retrieval_config,
)
from shopee_match.retrieval.vector_index import (
    ExactCosineIndex,
    FaissHnswIndex,
    search_result_to_ranking,
)
from shopee_match.training.multimodal_data import load_cached_multimodal_split
from shopee_match.training.multimodal_trainer import (
    _git_state,
    _resolve_device,
    extract_joint_embeddings,
)

LOGGER = logging.getLogger(__name__)
FloatArray = NDArray[np.float32]


def _write_text_atomic(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def _write_embeddings_atomic(
    path: Path,
    posting_ids: tuple[str, ...],
    embeddings: FloatArray,
    contract: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            posting_ids=np.asarray(posting_ids, dtype=str),
            embeddings=embeddings,
            contract=np.asarray([json.dumps(contract, sort_keys=True)], dtype=str),
        )
    temporary.replace(path)


def _load_phase6_model(
    config: CandidateRetrievalConfig, device: torch.device
) -> LearnedMultimodalFusion:
    checkpoint = torch.load(config.source.checkpoint_path, map_location="cpu", weights_only=False)
    phase6 = config.source.experiment
    expected_spec = asdict(phase6.source.experiment.model_spec)
    selection = config.source.metrics["selection"]
    if (
        checkpoint.get("checkpoint_version") != "phase6.hard_negative_finetune.v1"
        or checkpoint.get("model_spec") != expected_spec
        or checkpoint.get("seed") != config.seed
        or checkpoint.get("hard_negative_manifest_sha256") != config.source.mined_manifest_sha256
        or checkpoint.get("best_epoch") != selection["best_epoch"]
        or not np.isclose(
            float(checkpoint.get("best_metric", float("nan"))),
            float(selection["best_metric"]),
            rtol=0.0,
            atol=1e-12,
        )
    ):
        raise ConfigurationError("Frozen Phase 6 checkpoint does not match its accepted evidence")
    model = LearnedMultimodalFusion(phase6.source.experiment.model_spec)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model.to(device)


def _truncate_ranking(ranking: Ranking, candidate_k: int) -> Ranking:
    return {query_id: candidates[:candidate_k] for query_id, candidates in ranking.items()}


def _retrieval_curve(
    ranking: Ranking,
    label_by_id: dict[str, str],
    k_values: tuple[int, ...],
) -> dict[str, dict[str, float]]:
    return {
        str(candidate_k): retrieval_metrics(
            _truncate_ranking(ranking, candidate_k),
            label_by_id,
            (candidate_k,),
            candidate_k,
        )
        for candidate_k in k_values
    }


def _select_candidate_k(
    curve: dict[str, dict[str, float]], target_recall: float
) -> tuple[int, bool]:
    ordered = sorted(int(value) for value in curve)
    for candidate_k in ordered:
        if curve[str(candidate_k)][f"recall@{candidate_k}"] >= target_recall:
            return candidate_k, True
    return ordered[-1], False


def _candidate_agreement(exact: Ranking, approximate: Ranking, candidate_k: int) -> float:
    values: list[float] = []
    for query_id in exact:
        exact_ids = {row.posting_id for row in exact[query_id][:candidate_k]}
        approximate_ids = {row.posting_id for row in approximate[query_id][:candidate_k]}
        values.append(len(exact_ids & approximate_ids) / candidate_k)
    return float(np.mean(values))


def _search_index(
    index: ExactCosineIndex | FaissHnswIndex,
    embeddings: FloatArray,
    posting_ids: tuple[str, ...],
    candidate_k: int,
    block_size: int,
) -> tuple[NDArray[np.int64], FloatArray]:
    if isinstance(index, ExactCosineIndex):
        return index.search(
            embeddings,
            candidate_k,
            query_ids=posting_ids,
            block_size=block_size,
        )
    return index.search(embeddings, candidate_k, query_ids=posting_ids)


def _profile_index(
    index: ExactCosineIndex | FaissHnswIndex,
    embeddings: FloatArray,
    posting_ids: tuple[str, ...],
    candidate_k: int,
    *,
    block_size: int,
    query_count: int,
    repetitions: int,
) -> dict[str, float]:
    sample_count = min(query_count, len(posting_ids))
    sample_indices = np.linspace(0, len(posting_ids) - 1, sample_count, dtype=np.int64)
    for index_value in sample_indices[: min(16, sample_count)]:
        row = int(index_value)
        _search_index(
            index,
            embeddings[row : row + 1],
            (posting_ids[row],),
            candidate_k,
            block_size,
        )
    latencies_ms: list[float] = []
    for _repeat in range(repetitions):
        for index_value in sample_indices:
            row = int(index_value)
            started = time.perf_counter()
            _search_index(
                index,
                embeddings[row : row + 1],
                (posting_ids[row],),
                candidate_k,
                block_size,
            )
            latencies_ms.append((time.perf_counter() - started) * 1000)
    batch_started = time.perf_counter()
    _search_index(index, embeddings, posting_ids, candidate_k, block_size)
    batch_seconds = time.perf_counter() - batch_started
    values = np.asarray(latencies_ms, dtype=np.float64)
    return {
        "single_query_p50_ms": float(np.percentile(values, 50)),
        "single_query_p95_ms": float(np.percentile(values, 95)),
        "single_query_mean_ms": float(values.mean()),
        "profiled_queries": float(len(values)),
        "batch_search_seconds": batch_seconds,
        "batch_throughput_queries_per_second": len(posting_ids) / batch_seconds,
    }


def _process_rss_bytes() -> int | None:
    try:
        psutil = importlib.import_module("psutil")
        return int(psutil.Process().memory_info().rss)
    except (ImportError, AttributeError, OSError):
        return None


def _band_group_size(size: int) -> str:
    if size <= 2:
        return "2"
    if size <= 5:
        return "3_to_5"
    if size <= 9:
        return "6_to_9"
    return "10_plus"


def _band_title_length(length: int) -> str:
    if length <= 40:
        return "short_0_to_40"
    if length <= 80:
        return "medium_41_to_80"
    return "long_81_plus"


def _failure_analysis(
    ranking: Ranking,
    split: EvaluationSplit,
    candidate_k: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    labels = split.label_by_id
    item_by_id = {item.posting_id: item for item in split.items}
    members: dict[str, set[str]] = defaultdict(set)
    for posting_id, label in labels.items():
        members[label].add(posting_id)
    label_sizes = Counter(labels.values())
    strata: dict[str, dict[str, list[float]]] = {
        "group_size": defaultdict(list),
        "title_length": defaultdict(list),
        "positive_phash": defaultdict(list),
    }
    no_positive = partial = 0
    examples: list[dict[str, Any]] = []
    for query_id in sorted(ranking):
        positives = members[labels[query_id]] - {query_id}
        candidates = {row.posting_id for row in ranking[query_id][:candidate_k]}
        query_recall = len(positives & candidates) / len(positives)
        query_item = item_by_id[query_id]
        exact_phash = any(
            item_by_id[positive].image_phash == query_item.image_phash for positive in positives
        )
        keys = {
            "group_size": _band_group_size(label_sizes[labels[query_id]]),
            "title_length": _band_title_length(len(query_item.title)),
            "positive_phash": "has_exact_positive" if exact_phash else "no_exact_positive",
        }
        for dimension, key in keys.items():
            strata[dimension][key].append(query_recall)
        if query_recall == 0:
            no_positive += 1
        elif query_recall < 1:
            partial += 1
        if query_recall < 1 and len(examples) < 30:
            top = ranking[query_id][0]
            examples.append(
                {
                    "query_id": query_id,
                    "query_title": query_item.title,
                    "label_group": labels[query_id],
                    "group_size": label_sizes[labels[query_id]],
                    "query_recall": query_recall,
                    "has_exact_positive_phash": exact_phash,
                    "top_candidate_id": top.posting_id,
                    "top_candidate_title": item_by_id[top.posting_id].title,
                    "top_candidate_label": labels[top.posting_id],
                    "top_candidate_score": top.score,
                }
            )
    summary = {
        "queries": len(ranking),
        "no_positive_retrieved": no_positive,
        "partial_group_retrieval": partial,
        "complete_group_retrieval": len(ranking) - no_positive - partial,
        "strata": {
            dimension: {
                key: {
                    "queries": len(values),
                    "mean_recall": float(np.mean(values)),
                    "zero_recall_queries": sum(value == 0 for value in values),
                }
                for key, values in sorted(groups.items())
            }
            for dimension, groups in strata.items()
        },
    }
    return summary, examples


def _approximate_disagreements(
    exact: Ranking,
    approximate: Ranking,
    candidate_k: int,
    split: EvaluationSplit,
) -> list[dict[str, Any]]:
    item_by_id = {item.posting_id: item for item in split.items}
    rows: list[dict[str, Any]] = []
    for query_id in exact:
        exact_ids = {row.posting_id for row in exact[query_id][:candidate_k]}
        approximate_ids = {row.posting_id for row in approximate[query_id][:candidate_k]}
        agreement = len(exact_ids & approximate_ids) / candidate_k
        if agreement < 1:
            rows.append(
                {
                    "query_id": query_id,
                    "query_title": item_by_id[query_id].title,
                    "agreement": agreement,
                    "exact_only": sorted(exact_ids - approximate_ids),
                    "approximate_only": sorted(approximate_ids - exact_ids),
                }
            )
    rows.sort(key=lambda row: (row["agreement"], row["query_id"]))
    return rows[:30]


def _render_report(run: dict[str, Any]) -> str:
    exact = run["exact"]
    approximate = run["approximate"]
    selected_k = run["selection"]["candidate_k"]
    exact_rows = "\n".join(
        f"| {candidate_k} | {metrics[f'recall@{candidate_k}']:.5f} | "
        f"{metrics[f'hit_rate@{candidate_k}']:.5f} | {metrics[f'map@{candidate_k}']:.5f} |"
        for candidate_k, metrics in (
            (int(key), value) for key, value in exact["retrieval_curve"].items()
        )
    )
    faiss_rows = "\n".join(
        f"| {trial['ef_search']} | {trial['recall']:.5f} | {trial['recall_delta']:+.5f} | "
        f"{trial['exact_candidate_agreement']:.5f} | {str(trial['passes']).lower()} |"
        for trial in approximate["trials"]
    )
    exact_profile = exact["profile"]
    faiss_profile = approximate["selected_profile"]
    failure = exact["failure_analysis"]
    outcome = run["selection"]["status"]
    hnsw_settings = (
        f"`{approximate['m']}` / `{approximate['ef_construction']}` / "
        f"`{approximate['rerank_buffer']}`"
    )
    efficiency_rows = "\n".join(
        (
            "| Single-query p50 latency | "
            f"{exact_profile['single_query_p50_ms']:.3f} ms | "
            f"{faiss_profile['single_query_p50_ms']:.3f} ms |",
            "| Single-query p95 latency | "
            f"{exact_profile['single_query_p95_ms']:.3f} ms | "
            f"{faiss_profile['single_query_p95_ms']:.3f} ms |",
            "| Batch throughput | "
            f"{exact_profile['batch_throughput_queries_per_second']:.2f} queries/s | "
            f"{faiss_profile['batch_throughput_queries_per_second']:.2f} queries/s |",
            f"| Estimated in-memory index | {exact['estimated_memory_bytes']:,} bytes | "
            f"{approximate['estimated_memory_bytes']:,} bytes |",
            f"| Serialized index | {exact['serialized_bytes']:,} bytes | "
            f"{approximate['serialized_bytes']:,} bytes |",
        )
    )
    return f"""# Candidate retrieval benchmark

## Outcome

Phase 7 status: **{outcome}**. The benchmark uses only validation to select candidate K and FAISS
`efSearch`; held-out test data is not evaluated. Exact cosine search is the quality reference, and
FAISS HNSW is accepted only if it preserves recall and candidate-set agreement.

## Exact candidate ceiling

| K | Recall@K | Hit rate@K | mAP@K |
|---:|---:|---:|---:|
{exact_rows}

- Target macro candidate recall: `{run["selection"]["target_recall"]:.3f}`
- Selected exact K: `{selected_k}`
- Target reached: `{str(run["selection"]["target_reached"]).lower()}`

## FAISS HNSW sweep at selected K

| efSearch | Recall | Delta vs exact | Exact candidate agreement | Gate |
|---:|---:|---:|---:|---|
{faiss_rows}

- Selected `efSearch`: `{run["selection"]["ef_search"]}`
- Index: HNSW Flat, inner product over L2-normalized embeddings
- HNSW M / efConstruction / rerank buffer: {hnsw_settings}

## Efficiency

| Measure | Exact | FAISS HNSW |
|---|---:|---:|
{efficiency_rows}

- Embedding extraction: `{run["embedding"]["throughput_per_second"]:.2f}` listings/s
- Embedding matrix: `{run["embedding"]["storage_bytes"]:,}` bytes
- Embedding dimension: `{run["embedding"]["dimension"]}`

## Retrieval failures at exact K={selected_k}

| Category | Queries |
|---|---:|
| No positive candidate retrieved | {failure["no_positive_retrieved"]} |
| Some but not all group members retrieved | {failure["partial_group_retrieval"]} |
| Complete group retrieved | {failure["complete_group_retrieval"]} |

The local failure-review JSON contains bounded title-rich examples and approximate disagreements;
it remains ignored because it is generated evidence. Aggregate group-size, title-length, and
exact-positive-pHash strata are stored in metrics.

## Interpretation

Recall@K is the Phase 7 primary metric because a match omitted here cannot be recovered by the pair
classifier in Phase 8. Hit rate is less strict: it needs only one duplicate, while macro Recall@K
rewards retrieving the full product group. HNSW changes candidate generation only; it does not apply
the Phase 6 pair head or make match/no-match decisions.

## Limitations

- These are validation-only model/index-selection results, not a new held-out test result.
- The measured catalog has `{run["data"]["listings"]:,}` listings; latency and memory must be
  remeasured before claiming behavior at production catalog scale.
- Timings are hardware- and environment-specific. Quality gates, serialized round-trip checks, and
  exact agreement are the portable parts of this benchmark.
"""


def run_candidate_retrieval_benchmark(config_path: Path) -> dict[str, object]:
    """Extract validation embeddings, verify exact search, then benchmark FAISS HNSW."""
    config = load_candidate_retrieval_config(config_path)
    existing = [
        str(path) for path in (config.artifacts.metrics, config.artifacts.report) if path.exists()
    ]
    if existing:
        raise OutputConflictError(
            "Refusing to overwrite completed Phase 7 evidence: " + ", ".join(existing)
        )
    seed_everything(config.seed, deterministic=True)
    device = _resolve_device(config.embedding.device)
    phase6 = config.source.experiment
    source_experiment = phase6.source.experiment
    split = load_named_split(
        source_experiment.data.metadata_csv,
        source_experiment.data.split_manifest,
        "validation",
    )
    dataset = load_cached_multimodal_split(source_experiment, "validation")
    expected_ids = tuple(item.posting_id for item in split.items)
    if dataset.posting_ids != expected_ids:
        raise DataValidationError("Validation cache order differs from the validation manifest")
    loader = DataLoader(
        dataset,
        batch_size=config.embedding.batch_size,
        shuffle=False,
        num_workers=config.embedding.num_workers,
    )
    model = _load_phase6_model(config, device)
    posting_ids, _image, _text, embeddings, extraction_seconds = extract_joint_embeddings(
        model, loader, device
    )
    if posting_ids != expected_ids:
        raise DataValidationError("Extracted embeddings do not align with validation posting IDs")
    contract = {
        "version": "phase7.listing_embeddings.v1",
        "split": "validation",
        "checkpoint_sha256": config.source.checkpoint_sha256,
        "split_manifest_sha256": sha256_file(source_experiment.data.split_manifest),
        "normalized": True,
        "test_accessed": False,
    }
    _write_embeddings_atomic(
        config.artifacts.embedding_cache,
        posting_ids,
        embeddings,
        contract,
    )
    maximum_k = max(config.selection.k_values)
    rss_before = _process_rss_bytes()

    exact_build_started = time.perf_counter()
    exact_index = ExactCosineIndex(posting_ids, embeddings)
    exact_build_seconds = time.perf_counter() - exact_build_started
    exact_indices, exact_scores = exact_index.search(
        embeddings,
        maximum_k,
        query_ids=posting_ids,
        block_size=config.exact.block_size,
    )
    exact_ranking = search_result_to_ranking(posting_ids, posting_ids, exact_indices, exact_scores)
    exact_curve = _retrieval_curve(exact_ranking, split.label_by_id, config.selection.k_values)
    selected_k, target_reached = _select_candidate_k(exact_curve, config.selection.target_recall)
    exact_index.save(config.artifacts.exact_index)
    restored_exact = ExactCosineIndex.load(config.artifacts.exact_index)
    restored_indices, restored_scores = restored_exact.search(
        embeddings,
        maximum_k,
        query_ids=posting_ids,
        block_size=config.exact.block_size,
    )
    exact_round_trip = np.array_equal(exact_indices, restored_indices) and np.allclose(
        exact_scores, restored_scores, rtol=0.0, atol=1e-7
    )
    if not exact_round_trip:
        raise DataValidationError("Serialized exact index changed Top-K output")
    exact_selected_recall = exact_curve[str(selected_k)][f"recall@{selected_k}"]
    exact_profile = _profile_index(
        exact_index,
        embeddings,
        posting_ids,
        selected_k,
        block_size=config.exact.block_size,
        query_count=config.selection.latency_query_count,
        repetitions=config.selection.latency_repetitions,
    )
    exact_failure, exact_examples = _failure_analysis(exact_ranking, split, selected_k)

    faiss_build_started = time.perf_counter()
    faiss_index = FaissHnswIndex(
        posting_ids,
        embeddings,
        m=config.faiss.m,
        ef_construction=config.faiss.ef_construction,
        ef_search=config.faiss.ef_search_values[0],
        threads=config.faiss.threads,
        rerank_buffer=config.faiss.rerank_buffer,
    )
    faiss_build_seconds = time.perf_counter() - faiss_build_started
    approximate_trials: list[dict[str, Any]] = []
    ranking_by_ef: dict[int, Ranking] = {}
    for ef_search in config.faiss.ef_search_values:
        faiss_index.ef_search = ef_search
        approximate_indices, approximate_scores = faiss_index.search(
            embeddings, maximum_k, query_ids=posting_ids
        )
        ranking = search_result_to_ranking(
            posting_ids, posting_ids, approximate_indices, approximate_scores
        )
        ranking_by_ef[ef_search] = ranking
        selected_metrics = retrieval_metrics(
            _truncate_ranking(ranking, selected_k),
            split.label_by_id,
            (selected_k,),
            selected_k,
        )
        recall = selected_metrics[f"recall@{selected_k}"]
        agreement = _candidate_agreement(exact_ranking, ranking, selected_k)
        passes = (
            recall >= exact_selected_recall - config.selection.maximum_approximate_recall_drop
            and agreement >= config.selection.minimum_exact_candidate_agreement
        )
        approximate_trials.append(
            {
                "ef_search": ef_search,
                "recall": recall,
                "recall_delta": recall - exact_selected_recall,
                "exact_candidate_agreement": agreement,
                "hit_rate": selected_metrics[f"hit_rate@{selected_k}"],
                "map": selected_metrics[f"map@{selected_k}"],
                "passes": passes,
            }
        )
        LOGGER.info(
            "FAISS efSearch=%d recall@%d=%.5f agreement=%.5f pass=%s",
            ef_search,
            selected_k,
            recall,
            agreement,
            passes,
        )
    passing_trials = [trial for trial in approximate_trials if trial["passes"]]
    selected_trial = passing_trials[0] if passing_trials else approximate_trials[-1]
    selected_ef = int(selected_trial["ef_search"])
    selected_ranking = ranking_by_ef[selected_ef]
    faiss_index.ef_search = selected_ef
    faiss_index.save(config.artifacts.faiss_index, config.artifacts.faiss_metadata)
    restored_faiss = FaissHnswIndex.load(
        config.artifacts.faiss_index, config.artifacts.faiss_metadata
    )
    faiss_indices, faiss_scores = faiss_index.search(embeddings, maximum_k, query_ids=posting_ids)
    restored_faiss_indices, restored_faiss_scores = restored_faiss.search(
        embeddings, maximum_k, query_ids=posting_ids
    )
    faiss_round_trip = np.array_equal(faiss_indices, restored_faiss_indices) and np.allclose(
        faiss_scores, restored_faiss_scores, rtol=0.0, atol=1e-7
    )
    if not faiss_round_trip:
        raise DataValidationError("Serialized FAISS index changed Top-K output")
    faiss_profile = _profile_index(
        faiss_index,
        embeddings,
        posting_ids,
        selected_k,
        block_size=config.exact.block_size,
        query_count=config.selection.latency_query_count,
        repetitions=config.selection.latency_repetitions,
    )
    selected_approximate_curve = _retrieval_curve(
        selected_ranking, split.label_by_id, config.selection.k_values
    )
    disagreements = _approximate_disagreements(exact_ranking, selected_ranking, selected_k, split)
    _write_text_atomic(
        config.artifacts.review,
        json.dumps(
            {
                "version": "phase7.failure_review.v1",
                "split": "validation",
                "selected_k": selected_k,
                "exact_failures": exact_examples,
                "approximate_disagreements": disagreements,
                "test_accessed": False,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
    )
    rss_after = _process_rss_bytes()
    commit, dirty = _git_state()
    faiss_module = importlib.import_module("faiss")
    status = (
        "phase7_complete_validation_only"
        if target_reached and bool(selected_trial["passes"])
        else "phase7_exit_gate_failed"
    )
    run: dict[str, Any] = {
        "pipeline_version": "phase7.candidate_retrieval.v1",
        "status": status,
        "provenance": {
            "config_sha256": canonical_text_sha256(config.config_path),
            "phase6_config_sha256": config.source.phase6_config_sha256,
            "phase6_checkpoint_sha256": config.source.checkpoint_sha256,
            "phase6_metrics_sha256": config.source.metrics_sha256,
            "mined_manifest_sha256": config.source.mined_manifest_sha256,
            "split_manifest_sha256": sha256_file(source_experiment.data.split_manifest),
            "git_commit": commit,
            "git_dirty": dirty,
            "seed": config.seed,
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "faiss": str(faiss_module.__version__),
            "device": str(device),
        },
        "data": {
            "split": "validation",
            "listings": len(posting_ids),
            "test_accessed": False,
        },
        "embedding": {
            "dimension": int(embeddings.shape[1]),
            "normalized": bool(np.allclose(np.linalg.norm(embeddings, axis=1), 1.0, atol=1e-5)),
            "extraction_seconds": extraction_seconds,
            "throughput_per_second": len(posting_ids) / extraction_seconds,
            "storage_bytes": int(embeddings.nbytes),
            "cache_file_bytes": config.artifacts.embedding_cache.stat().st_size,
        },
        "selection": {
            "target_recall": config.selection.target_recall,
            "candidate_k": selected_k,
            "target_reached": target_reached,
            "ef_search": selected_ef,
            "status": status,
        },
        "exact": {
            "backend": exact_index.backend,
            "retrieval_curve": exact_curve,
            "selected_recall": exact_selected_recall,
            "build_seconds": exact_build_seconds,
            "profile": exact_profile,
            "estimated_memory_bytes": exact_index.estimated_memory_bytes,
            "serialized_bytes": config.artifacts.exact_index.stat().st_size,
            "round_trip_identical": exact_round_trip,
            "failure_analysis": exact_failure,
        },
        "approximate": {
            "backend": faiss_index.backend,
            "m": config.faiss.m,
            "ef_construction": config.faiss.ef_construction,
            "rerank_buffer": config.faiss.rerank_buffer,
            "threads": config.faiss.threads,
            "trials": approximate_trials,
            "selected_curve": selected_approximate_curve,
            "selected_profile": faiss_profile,
            "build_seconds": faiss_build_seconds,
            "estimated_memory_bytes": faiss_index.estimated_memory_bytes,
            "serialized_bytes": (
                config.artifacts.faiss_index.stat().st_size
                + config.artifacts.faiss_metadata.stat().st_size
            ),
            "round_trip_identical": faiss_round_trip,
            "selected_passes": bool(selected_trial["passes"]),
            "disagreement_examples": len(disagreements),
        },
        "process_memory": {
            "rss_before_index_bytes": rss_before,
            "rss_after_indexes_bytes": rss_after,
            "rss_delta_bytes": (
                rss_after - rss_before if rss_before is not None and rss_after is not None else None
            ),
        },
        "artifacts": {
            "embedding_cache": str(config.artifacts.embedding_cache),
            "exact_index": str(config.artifacts.exact_index),
            "faiss_index": str(config.artifacts.faiss_index),
            "faiss_metadata": str(config.artifacts.faiss_metadata),
            "failure_review": str(config.artifacts.review),
        },
        "test": {"status": "disabled_phase7_validation_only"},
    }
    _write_text_atomic(config.artifacts.metrics, json.dumps(run, indent=2, sort_keys=True) + "\n")
    _write_text_atomic(config.artifacts.report, _render_report(run))
    return {
        "status": status,
        "selected_k": selected_k,
        "selected_ef_search": selected_ef,
        "exact_recall": exact_selected_recall,
        "approximate_recall": selected_trial["recall"],
        "exact_candidate_agreement": selected_trial["exact_candidate_agreement"],
        "metrics": str(config.artifacts.metrics),
        "report": str(config.artifacts.report),
        "test_accessed": False,
    }
