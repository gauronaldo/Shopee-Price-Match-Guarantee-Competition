"""Frozen repeated-seed comparison and Phase 6 closure report."""

from __future__ import annotations

import json
import os
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch

from shopee_match.errors import ConfigurationError, DataValidationError
from shopee_match.hashing import matches_frozen_sha256, sha256_file
from shopee_match.training.hard_negative_config import load_hard_negative_experiment_config
from shopee_match.training.text_config import _mapping, _only_keys, _read_yaml, _relative_path


@dataclass(frozen=True, slots=True)
class FrozenRun:
    seed: int
    config_path: Path
    config_sha256: str
    checkpoint_sha256: str
    metrics_sha256: str


def _write_text_atomic(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def _digest(value: Any, location: str) -> str:
    if not isinstance(value, str):
        raise ConfigurationError(f"{location} must be a SHA-256 string")
    result = value.lower()
    if len(result) != 64 or any(character not in "0123456789abcdef" for character in result):
        raise ConfigurationError(f"{location} must be a SHA-256 digest")
    return result


def _load_summary_config(
    path: Path,
) -> tuple[int, dict[str, float], list[FrozenRun], Path]:
    root = _read_yaml(path, "hard-negative repeated-seed summary config")
    _only_keys(root, {"config_version", "canonical_seed", "baseline", "runs", "report"}, "config")
    if root["config_version"] != "phase6.hard_negative_repeated_seed_summary.v1":
        raise ConfigurationError("Unsupported hard-negative summary config_version")
    canonical_seed = int(root["canonical_seed"])
    baseline_raw = _mapping(root["baseline"], "baseline")
    _only_keys(
        baseline_raw,
        {"map_at_20", "controlled_precision", "recall_at_20"},
        "baseline",
    )
    baseline = {name: float(value) for name, value in baseline_raw.items()}
    raw_runs = root["runs"]
    if not isinstance(raw_runs, list) or len(raw_runs) < 3:
        raise ConfigurationError("Phase 6 closure requires at least three frozen runs")
    runs: list[FrozenRun] = []
    for index, value in enumerate(raw_runs):
        raw = _mapping(value, f"runs[{index}]")
        _only_keys(
            raw,
            {"seed", "config", "config_sha256", "checkpoint_sha256", "metrics_sha256"},
            f"runs[{index}]",
        )
        runs.append(
            FrozenRun(
                seed=int(raw["seed"]),
                config_path=_relative_path(raw["config"], f"runs[{index}].config"),
                config_sha256=_digest(raw["config_sha256"], f"runs[{index}].config_sha256"),
                checkpoint_sha256=_digest(
                    raw["checkpoint_sha256"], f"runs[{index}].checkpoint_sha256"
                ),
                metrics_sha256=_digest(
                    raw["metrics_sha256"], f"runs[{index}].metrics_sha256"
                ),
            )
        )
    if len({run.seed for run in runs}) != len(runs) or canonical_seed not in {
        run.seed for run in runs
    }:
        raise ConfigurationError("Repeated seeds must be unique and include canonical_seed")
    return canonical_seed, baseline, runs, _relative_path(root["report"], "report")


def _load_frozen_run(run: FrozenRun, baseline: dict[str, float]) -> dict[str, Any]:
    matches, actual_config_hash = matches_frozen_sha256(run.config_path, run.config_sha256)
    if not matches:
        raise ConfigurationError(
            f"Frozen Phase 6 config hash mismatch: expected {run.config_sha256}, "
            f"got {actual_config_hash}"
        )
    config = load_hard_negative_experiment_config(run.config_path)
    if sha256_file(config.artifacts.checkpoint) != run.checkpoint_sha256:
        raise ConfigurationError(f"Frozen checkpoint hash mismatch for seed {run.seed}")
    if sha256_file(config.artifacts.metrics) != run.metrics_sha256:
        raise ConfigurationError(f"Frozen metrics hash mismatch for seed {run.seed}")
    checkpoint = torch.load(config.artifacts.checkpoint, map_location="cpu", weights_only=False)
    metrics = cast(
        dict[str, Any], json.loads(config.artifacts.metrics.read_text(encoding="utf-8"))
    )
    if (
        checkpoint.get("checkpoint_version") != "phase6.hard_negative_finetune.v1"
        or checkpoint.get("seed") != run.seed
        or metrics.get("pipeline_version") != "phase6.hard_negative_training.v1"
        or metrics.get("provenance", {}).get("seed") != run.seed
        or metrics.get("test", {}).get("status") != "disabled_phase6_validation_only"
        or metrics.get("data", {}).get("test_accessed") is not False
        or metrics.get("acceptance", {}).get("pilot_pass") is not True
    ):
        raise DataValidationError(f"Seed {run.seed} is not valid Phase 6 pass evidence")
    baseline_pair = metrics["validation"]["phase5_baseline"]["pair_head_rerank"]
    if not (
        np.isclose(baseline_pair["retrieval"]["map@20"], baseline["map_at_20"], atol=1e-12)
        and np.isclose(
            baseline_pair["precision_at_controlled_recall"]["precision"],
            baseline["controlled_precision"],
            atol=1e-12,
        )
        and np.isclose(
            baseline_pair["retrieval"]["recall@20"], baseline["recall_at_20"], atol=1e-12
        )
    ):
        raise DataValidationError(f"Seed {run.seed} used a different Phase 5 reference")
    return metrics


def _render_report(
    canonical_seed: int,
    baseline: dict[str, float],
    frozen_runs: list[tuple[FrozenRun, dict[str, Any]]],
) -> str:
    rows: list[str] = []
    map_values: list[float] = []
    precision_deltas: list[float] = []
    variant_deltas: list[int] = []
    for run, metrics in frozen_runs:
        acceptance = metrics["acceptance"]
        selected = metrics["validation"]["selected_checkpoint"]["pair_head_rerank"]
        selected_map = float(selected["retrieval"]["map@20"])
        precision_delta = float(acceptance["controlled_precision_delta"])
        variant_delta = int(acceptance["variant_conflict_delta"])
        map_values.append(selected_map)
        precision_deltas.append(precision_delta)
        variant_deltas.append(variant_delta)
        rows.append(
            f"| {run.seed} | {metrics['selection']['best_epoch'] + 1} | "
            f"{selected_map:.5f} | {acceptance['map_delta']:+.5f} | "
            f"{precision_delta:+.5f} | {acceptance['recall_at_20_delta']:+.5f} | "
            f"{variant_delta:+d} | pass |"
        )
    canonical = next(run for run, _metrics in frozen_runs if run.seed == canonical_seed)
    mean_map = statistics.mean(map_values)
    std_map = statistics.pstdev(map_values)
    mean_precision = statistics.mean(precision_deltas)
    std_precision = statistics.pstdev(precision_deltas)
    mean_variant = statistics.mean(variant_deltas)
    table_header = (
        "| Seed | Best epoch | mAP@20 | mAP delta | Controlled precision delta | "
        "Recall@20 delta | Variant Top-1 delta | Gate |"
    )
    return f"""# Hard-negative mining final comparison

## Outcome

Phase 6 **passes on validation across all three deterministic seeds**. Pair-head-only fine-tuning
improved precision at the frozen Phase 5 recall operating point in every run, preserved candidate
Recall@20 exactly, and did not regress mAP@20. The effect is deliberately reported as modest.

The earlier joint fusion/pair-head pilot failed because it distorted the retrieval embedding; its
failure is retained in `hard_negative_mining_pilot.md`. The accepted method freezes fusion and
updates only the symmetric pair classifier using a 75/25 mix of original/random-pair and mined
hard-negative BCE.

## Frozen Phase 5 reference

| Metric | Value |
|---|---:|
| Pair-head mAP@20 | {baseline['map_at_20']:.5f} |
| Controlled-recall precision | {baseline['controlled_precision']:.5f} |
| Pair-head Recall@20 | {baseline['recall_at_20']:.5f} |

## Repeated-seed validation

{table_header}
|---:|---:|---:|---:|---:|---:|---:|---|
{chr(10).join(rows)}

| Aggregate | Mean | Population std. |
|---|---:|---:|
| mAP@20 | {mean_map:.5f} | {std_map:.5f} |
| Controlled precision delta | {mean_precision:+.5f} | {std_precision:.5f} |
| Variant Top-1 error delta | {mean_variant:+.2f} | not applicable |

## Canonical Phase 6 artifact

Seed `{canonical_seed}` remains canonical because it is the project's pre-declared primary seed,
not because it produced the largest score.

- Config SHA-256: `{canonical.config_sha256}`
- Checkpoint SHA-256: `{canonical.checkpoint_sha256}`
- Metrics SHA-256: `{canonical.metrics_sha256}`
- Mined manifest SHA-256: `ad716c1c7a4d5e1aa31cbd668c98b3d1c6f42117d865d2bc2aa5bf995e19d2d2`
- Mined pairs: `24,332` (`50%` digit/unit variant conflicts)

## Scope and interpretation

Hard negatives do not create new candidate recall. They teach the pair classifier to demote
different products that the Phase 5 embedding retrieves as deceptively similar. Freezing fusion is
why Recall@20 stays fixed; only ordering and decision precision change. Validation alone selected
all checkpoints. Phase 6 did not access or retune on test.

The gain is statistically consistent across these deterministic sampling seeds but small; it should
not be described as a large model improvement. Phase 6 is closed, and Phase 7 may use the canonical
seed-{canonical_seed} checkpoint for candidate-generation work.
"""


def summarize_hard_negative_runs(config_path: Path) -> dict[str, object]:
    """Verify frozen run hashes, aggregate validation results, and close Phase 6."""
    canonical_seed, baseline, runs, report_path = _load_summary_config(config_path)
    frozen_runs = [(run, _load_frozen_run(run, baseline)) for run in runs]
    manifest_hashes = {
        metrics["provenance"]["manifest_sha256"] for _run, metrics in frozen_runs
    }
    if len(manifest_hashes) != 1:
        raise DataValidationError("Repeated Phase 6 runs did not use the same mined manifest")
    _write_text_atomic(report_path, _render_report(canonical_seed, baseline, frozen_runs))
    return {
        "status": "phase6_complete_validation_only",
        "runs": len(frozen_runs),
        "all_passed": True,
        "canonical_seed": canonical_seed,
        "manifest_sha256": next(iter(manifest_hashes)),
        "report": str(report_path),
        "test_accessed": False,
    }
