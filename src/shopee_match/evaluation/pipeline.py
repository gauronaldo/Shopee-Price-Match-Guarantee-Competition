"""Orchestration for leakage-safe classical retrieval benchmarks."""

from __future__ import annotations

import hashlib
import json
import logging
import platform
import subprocess
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, ClassVar, cast

import cv2
import numpy as np

from shopee_match.evaluation.config import (
    ClassicalRetrievalConfig,
    load_classical_retrieval_config,
)
from shopee_match.evaluation.protocol import (
    EvaluationSplit,
    Ranking,
    load_splits,
    pair_metrics_at_threshold,
    retrieval_metrics,
    select_threshold,
)
from shopee_match.evaluation.report import render_report, render_threshold_svg
from shopee_match.features.image import rank_phash, rerank_orb
from shopee_match.features.text import CharTfidfModel
from shopee_match.retrieval.fusion import candidate_union, fuse_rankings

LOGGER = logging.getLogger(__name__)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_state() -> tuple[str, bool]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
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


def _peak_working_set_bytes() -> int | None:
    """Read peak resident memory without instrumenting or slowing the experiment."""
    if platform.system() == "Windows":
        import ctypes

        class ProcessMemoryCounters(ctypes.Structure):
            _fields_: ClassVar[list[tuple[str, Any]]] = [
                ("cb", ctypes.c_ulong),
                ("PageFaultCount", ctypes.c_ulong),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        counters = ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(counters)
        windows_api = cast(Any, ctypes).windll
        current_process = windows_api.kernel32.GetCurrentProcess
        current_process.restype = ctypes.c_void_p
        get_memory_info = windows_api.psapi.GetProcessMemoryInfo
        get_memory_info.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ProcessMemoryCounters),
            ctypes.c_ulong,
        ]
        get_memory_info.restype = ctypes.c_int
        succeeded = get_memory_info(current_process(), ctypes.byref(counters), counters.cb)
        return int(counters.PeakWorkingSetSize) if succeeded else None
    status = Path("/proc/self/status")
    if status.exists():
        for line in status.read_text(encoding="utf-8").splitlines():
            if line.startswith("VmHWM:"):
                return int(line.split()[1]) * 1024
    return None


def _timed(function: Callable[[], Ranking]) -> tuple[Ranking, float]:
    start = time.perf_counter()
    value = function()
    return value, time.perf_counter() - start


def _evaluate(
    validation: Ranking,
    test: Ranking,
    validation_split: EvaluationSplit,
    test_split: EvaluationSplit,
    config: ClassicalRetrievalConfig,
    runtime_seconds: float,
) -> dict[str, Any]:
    selected = select_threshold(validation, validation_split.label_by_id)
    return {
        "selected_threshold": selected["threshold"],
        "runtime_seconds": runtime_seconds,
        "validation": {
            "retrieval": retrieval_metrics(
                validation,
                validation_split.label_by_id,
                config.evaluation.recall_at,
                config.evaluation.average_precision_at,
            ),
            "pair": selected,
        },
        "test": {
            "retrieval": retrieval_metrics(
                test,
                test_split.label_by_id,
                config.evaluation.recall_at,
                config.evaluation.average_precision_at,
            ),
            "pair": pair_metrics_at_threshold(test, test_split.label_by_id, selected["threshold"]),
        },
    }


def _threshold_curve(ranking: Ranking, labels: dict[str, str]) -> list[dict[str, float]]:
    return [pair_metrics_at_threshold(ranking, labels, threshold / 40) for threshold in range(41)]


def _review_examples(
    ranking: Ranking, evaluation_split: EvaluationSplit, limit: int = 10
) -> dict[str, list[dict[str, Any]]]:
    by_id = {item.posting_id: item for item in evaluation_split.items}
    labels = evaluation_split.label_by_id
    result: dict[str, list[dict[str, Any]]] = {"successes": [], "failures": []}
    for query_id in sorted(ranking):
        if not ranking[query_id]:
            continue
        candidate = ranking[query_id][0]
        bucket = "successes" if labels[query_id] == labels[candidate.posting_id] else "failures"
        if len(result[bucket]) >= limit:
            continue
        result[bucket].append(
            {
                "query_id": query_id,
                "query_title": by_id[query_id].title,
                "candidate_id": candidate.posting_id,
                "candidate_title": by_id[candidate.posting_id].title,
                "score": candidate.score,
                "same_label": labels[query_id] == labels[candidate.posting_id],
            }
        )
    return result


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def run_classical_retrieval_benchmark(config_path: Path) -> dict[str, Any]:
    """Benchmark classical retrieval with validation-only model selection."""
    config = load_classical_retrieval_config(config_path)
    splits = load_splits(config.metadata_csv, config.split_manifest)
    train = splits["train"]
    validation = splits[config.evaluation.tune_split]
    test = splits[config.evaluation.final_split]

    LOGGER.info("Running supplied-pHash retrieval")
    phash_validation, phash_validation_time = _timed(
        lambda: rank_phash(validation.items, config.phash.candidate_k)
    )
    phash_test, phash_test_time = _timed(lambda: rank_phash(test.items, config.phash.candidate_k))

    LOGGER.info("Fitting train-only character TF-IDF and retrieving candidates")
    fit_start = time.perf_counter()
    text_model = CharTfidfModel.fit(
        train.items, config.tfidf.ngram_range, config.tfidf.max_features
    )
    text_fit_time = time.perf_counter() - fit_start
    tfidf_validation, tfidf_validation_time = _timed(
        lambda: text_model.rank(validation.items, config.tfidf.candidate_k)
    )
    tfidf_test, tfidf_test_time = _timed(
        lambda: text_model.rank(test.items, config.tfidf.candidate_k)
    )

    LOGGER.info("Reranking label-blind candidate unions with ORB")
    orb_candidates_validation = candidate_union(
        (phash_validation, tfidf_validation), config.orb.candidate_k_per_source
    )
    orb_candidates_test = candidate_union(
        (phash_test, tfidf_test), config.orb.candidate_k_per_source
    )
    orb_validation, orb_validation_time = _timed(
        lambda: rerank_orb(
            validation.items,
            orb_candidates_validation,
            config.image_dir,
            config.orb.features,
            config.orb.top_k,
        )
    )
    orb_test, orb_test_time = _timed(
        lambda: rerank_orb(
            test.items,
            orb_candidates_test,
            config.image_dir,
            config.orb.features,
            config.orb.top_k,
        )
    )

    LOGGER.info("Selecting late-fusion weight on validation")
    fusion_candidates = []
    fusion_tuning: dict[str, float] = {}
    fusion_tune_start = time.perf_counter()
    metric_name = f"map@{config.evaluation.average_precision_at}"
    for weight in config.fusion.weight_grid:
        ranking = fuse_rankings(phash_validation, tfidf_validation, weight, config.fusion.top_k)
        score = retrieval_metrics(
            ranking,
            validation.label_by_id,
            config.evaluation.recall_at,
            config.evaluation.average_precision_at,
        )[metric_name]
        fusion_candidates.append((score, -abs(weight - 0.5), -weight, weight, ranking))
        fusion_tuning[f"{weight:.6g}"] = score
    _score, _balance, _weight_tie_break, selected_weight, fusion_validation = max(
        fusion_candidates, key=lambda item: item[:3]
    )
    fusion_test = fuse_rankings(phash_test, tfidf_test, selected_weight, config.fusion.top_k)
    fusion_time = time.perf_counter() - fusion_tune_start

    phash_runtime = phash_validation_time + phash_test_time
    tfidf_runtime = text_fit_time + tfidf_validation_time + tfidf_test_time
    orb_stage_runtime = orb_validation_time + orb_test_time
    git_commit, git_dirty = _git_state()
    results: dict[str, Any] = {
        "pipeline_version": "classical_retrieval.benchmark_pipeline.v1",
        "provenance": {
            "config_version": config.config_version,
            "config_sha256": _sha256(config.config_path),
            "manifest_sha256": _sha256(config.split_manifest),
            "metadata_sha256": _sha256(config.metadata_csv),
            "git_commit": git_commit,
            "git_dirty": git_dirty,
            "seed": config.seed,
            "python": platform.python_version(),
            "opencv": cv2.__version__,
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
        "data": {name: len(value.items) for name, value in splits.items()},
        "evaluation": {
            "recall_at": list(config.evaluation.recall_at),
            "average_precision_at": config.evaluation.average_precision_at,
        },
        "selection": {
            "fusion_metric": metric_name,
            "fusion_text_weight": selected_weight,
            "fusion_validation_grid": fusion_tuning,
        },
        "model": {
            "tfidf_vocabulary_size": len(text_model.vocabulary),
            "tfidf_ngram_range": list(text_model.ngram_range),
            "orb_features": config.orb.features,
            "orb_candidate_k_per_source": config.orb.candidate_k_per_source,
        },
        "baselines": {
            "phash": _evaluate(
                phash_validation,
                phash_test,
                validation,
                test,
                config,
                phash_runtime,
            ),
            "tfidf": _evaluate(
                tfidf_validation,
                tfidf_test,
                validation,
                test,
                config,
                tfidf_runtime,
            ),
            "orb": _evaluate(
                orb_validation,
                orb_test,
                validation,
                test,
                config,
                phash_runtime + tfidf_runtime + orb_stage_runtime,
            ),
            "fusion": _evaluate(
                fusion_validation,
                fusion_test,
                validation,
                test,
                config,
                phash_runtime + tfidf_runtime + fusion_time,
            ),
        },
    }

    results["baselines"]["phash"]["stage_runtime_seconds"] = phash_runtime
    results["baselines"]["tfidf"]["stage_runtime_seconds"] = tfidf_runtime
    results["baselines"]["orb"]["stage_runtime_seconds"] = orb_stage_runtime
    results["baselines"]["fusion"]["stage_runtime_seconds"] = fusion_time

    curves = {
        "pHash": _threshold_curve(phash_validation, validation.label_by_id),
        "TF-IDF": _threshold_curve(tfidf_validation, validation.label_by_id),
        "ORB": _threshold_curve(orb_validation, validation.label_by_id),
        "Fusion": _threshold_curve(fusion_validation, validation.label_by_id),
    }
    query_count = len(validation.items) + len(test.items)
    results["efficiency"] = {
        "evaluation_queries": query_count,
        "process_peak_working_set_bytes": _peak_working_set_bytes(),
        "runtime_definition": (
            "Wall time for validation plus test; end-to-end ORB/fusion includes their pHash and "
            "TF-IDF candidate stages. Peak working set includes Python and native allocations."
        ),
        "mean_end_to_end_ms_per_query": {
            name: 1000 * float(result["runtime_seconds"]) / query_count
            for name, result in results["baselines"].items()
        },
    }
    metrics_path = config.artifacts.root / "metrics.json"
    examples_path = config.artifacts.root / "review_examples.json"
    _write_text(
        metrics_path, json.dumps(results, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )
    _write_text(
        examples_path,
        json.dumps(
            {
                name: _review_examples(ranking, test)
                for name, ranking in {
                    "phash": phash_test,
                    "tfidf": tfidf_test,
                    "orb": orb_test,
                    "fusion": fusion_test,
                }.items()
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
    )
    _write_text(config.artifacts.threshold_figure, render_threshold_svg(curves))
    _write_text(
        config.artifacts.report,
        render_report(results, config.artifacts.threshold_figure),
    )
    return {
        "status": "complete",
        "metrics": str(metrics_path),
        "report": str(config.artifacts.report),
        "threshold_figure": str(config.artifacts.threshold_figure),
        "selected_fusion_text_weight": selected_weight,
    }
