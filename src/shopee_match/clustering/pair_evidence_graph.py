"""Select safe graph thresholds for frozen recalibrated pair-evidence scores."""

from __future__ import annotations

import json
import logging
import platform
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, cast

import numpy as np

from shopee_match.clustering.benchmark import (
    _failure_review,
    _write_assignments_atomic,
    _write_text_atomic,
)
from shopee_match.clustering.pair_evidence_graph_config import load_pair_evidence_graph_config
from shopee_match.clustering.recall_recovery import (
    load_scored_pairs_file,
    sweep_recovery_policies,
)
from shopee_match.errors import OutputConflictError
from shopee_match.evaluation.protocol import load_splits
from shopee_match.hashing import canonical_text_sha256, sha256_file
from shopee_match.reproducibility import seed_everything
from shopee_match.training.multimodal_trainer import _git_state

LOGGER = logging.getLogger(__name__)


def _render_report(run: dict[str, Any]) -> str:
    selected = run["selection"]["selected"]
    cluster = selected["clustering"]
    pairwise = cluster["pairwise"]
    policy = selected["singleton_attachment"]
    checks = "\n".join(
        f"| {name} | `{str(passed).lower()}` |"
        for name, passed in selected["acceptance_checks"].items()
    )
    return f"""# Pair-Evidence Graph Selection

Status: **{run["status"]}**. This run reuses a frozen pair-evidence checkpoint and rescored
validation cache. It does not retrain a model or access test data.

| Validation clustering metric | Value |
|---|---:|
| Pairwise precision | {pairwise["precision"]:.5f} |
| Pairwise recall | {pairwise["recall"]:.5f} |
| Pairwise F1 | {pairwise["f1"]:.5f} |
| B-cubed F1 | {cluster["b_cubed"]["f1"]:.5f} |
| False-merge pair rate | {cluster["false_merge_pair_rate"]:.5f} |
| False-split group rate | {cluster["false_split_group_rate"]:.5f} |

Selected core threshold/rank/coverage: `{selected["pair_probability_threshold"]:.2f}` /
`{selected["reciprocal_rank"]}` / `{selected["cross_component_minimum_coverage"]:.2f}`.
Singleton attachment is `{str(policy["enabled"]).lower()}` with threshold
`{policy["probability_threshold"]}`, rank `{policy["reciprocal_rank"]}`, and
`{policy["minimum_support"]}` independent supports.

## Acceptance checks

| Check | Passed |
|---|---:|
{checks}

## Reproduction

```powershell
.venv\\Scripts\\shopee-entity-resolution select-pair-evidence-graph `
  --config configs\\experiment\\pair_evidence_graph_selection.yaml
```
"""


def run_pair_evidence_graph_selection(config_path: Path) -> dict[str, object]:
    """Select validation graph thresholds for a frozen evidence-score cache."""
    config = load_pair_evidence_graph_config(config_path)
    outputs = (
        config.artifacts.assignments,
        config.artifacts.metrics,
        config.artifacts.review,
        config.artifacts.report,
    )
    existing = [str(path) for path in outputs if path.exists()]
    if existing:
        raise OutputConflictError(
            "Refusing to overwrite pair-evidence graph evidence: " + ", ".join(existing)
        )
    started = time.perf_counter()
    seed_everything(config.seed, deterministic=True)
    experiment = config.source.experiment
    recovery = experiment.recovery
    entity = recovery.source.experiment
    phase7 = entity.source.experiment
    phase6 = phase7.source.experiment
    phase5 = phase6.source.experiment
    split = load_splits(phase5.data.metadata_csv, phase5.data.split_manifest)["validation"]
    posting_ids = tuple(item.posting_id for item in split.items)
    expected_pairs = int(recovery.source.metrics["pair_scoring"]["unique_pairs"])
    pairs = load_scored_pairs_file(
        config.source.scored_pairs_path,
        posting_ids,
        expected_count=expected_pairs,
    )
    selection_config = replace(recovery, selection=config.selection)
    LOGGER.info(
        "Loaded %d frozen evidence scores; selecting calibrated validation graph policy",
        len(pairs),
    )
    trials, selected, assignments = sweep_recovery_policies(
        selection_config, posting_ids, pairs, split.label_by_id
    )
    review = _failure_review(
        split,
        assignments,
        example_limit=config.selection.failure_example_limit,
    )
    status = (
        "pair_evidence_graph_target_reached_validation_only"
        if selected["passes_quality_target"]
        else "pair_evidence_graph_target_not_reached_validation_only"
    )
    commit, dirty = _git_state()
    run: dict[str, Any] = {
        "pipeline_version": "pair_evidence.graph_selection.v1",
        "status": status,
        "provenance": {
            "config_sha256": canonical_text_sha256(config.config_path),
            "evidence_config_sha256": config.source.evidence_config_sha256,
            "evidence_metrics_sha256": config.source.evidence_metrics_sha256,
            "checkpoint_sha256": config.source.checkpoint_sha256,
            "scored_pairs_sha256": config.source.scored_pairs_sha256,
            "split_manifest_sha256": sha256_file(phase5.data.split_manifest),
            "git_commit": commit,
            "git_dirty": dirty,
            "seed": config.seed,
            "python": platform.python_version(),
            "numpy": np.__version__,
        },
        "data": {
            "split": "validation",
            "listings": len(posting_ids),
            "pairs": len(pairs),
            "test_accessed": False,
            "model_retrained": False,
        },
        "selection": {
            "objective": "maximize_clustering_quality_subject_to_precision_and_false_merge_safety",
            "acceptance": asdict(config.selection.acceptance),
            "selected": selected,
            "trials": trials,
        },
        "failure_analysis": review["counts"],
        "runtime_seconds": time.perf_counter() - started,
        "test": {"status": "disabled_pair_evidence_graph_validation_only"},
    }
    _write_assignments_atomic(config.artifacts.assignments, assignments)
    _write_text_atomic(config.artifacts.review, json.dumps(review, indent=2, sort_keys=True) + "\n")
    _write_text_atomic(config.artifacts.metrics, json.dumps(run, indent=2, sort_keys=True) + "\n")
    _write_text_atomic(config.artifacts.report, _render_report(run))
    cluster = cast(dict[str, Any], selected["clustering"])
    return {
        "status": status,
        "pairwise_precision": cluster["pairwise"]["precision"],
        "pairwise_recall": cluster["pairwise"]["recall"],
        "pairwise_f1": cluster["pairwise"]["f1"],
        "b_cubed_f1": cluster["b_cubed"]["f1"],
        "false_merge_pair_rate": cluster["false_merge_pair_rate"],
        "false_split_group_rate": cluster["false_split_group_rate"],
        "metrics": str(config.artifacts.metrics),
        "report": str(config.artifacts.report),
        "test_accessed": False,
        "model_retrained": False,
    }
