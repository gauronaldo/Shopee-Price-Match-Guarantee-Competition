"""Validation-only recovery of conservative entity-resolution recall."""

from __future__ import annotations

import json
import logging
import math
import platform
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

import numpy as np

from shopee_match.clustering.benchmark import (
    _failure_review,
    _write_assignments_atomic,
    _write_text_atomic,
)
from shopee_match.clustering.graph import ClusterAssignment, ScoredPair, build_conservative_clusters
from shopee_match.clustering.metrics import clustering_metrics, group_size_strata
from shopee_match.clustering.recovery_config import (
    EntityRecallRecoveryConfig,
    RecoveryAcceptanceConfig,
    SingletonAttachmentPolicy,
    load_entity_recall_recovery_config,
)
from shopee_match.errors import DataValidationError, OutputConflictError
from shopee_match.evaluation.protocol import load_named_split
from shopee_match.hashing import canonical_text_sha256, sha256_file
from shopee_match.reproducibility import seed_everything
from shopee_match.training.multimodal_trainer import _git_state

LOGGER = logging.getLogger(__name__)


def load_scored_pairs_file(
    path: Path, posting_ids: tuple[str, ...], *, expected_count: int
) -> list[ScoredPair]:
    """Load and validate one frozen scored-pair JSONL cache."""
    index_by_id = {posting_id: index for index, posting_id in enumerate(posting_ids)}
    result: list[ScoredPair] = []
    seen: set[tuple[str, str]] = set()
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                payload = cast(dict[str, Any], json.loads(line))
                expected_keys = {
                    "left_posting_id",
                    "right_posting_id",
                    "cosine_similarity",
                    "pair_probability",
                    "left_rank",
                    "right_rank",
                    "variant_conflict",
                }
                if set(payload) != expected_keys:
                    raise DataValidationError(
                        f"Scored-pair row {line_number} does not match the frozen schema"
                    )
                left_id = str(payload["left_posting_id"])
                right_id = str(payload["right_posting_id"])
                if left_id not in index_by_id or right_id not in index_by_id or left_id == right_id:
                    raise DataValidationError(
                        f"Scored-pair row {line_number} references invalid posting IDs"
                    )
                key = (left_id, right_id) if left_id < right_id else (right_id, left_id)
                if key in seen:
                    raise DataValidationError(
                        f"Duplicate undirected scored pair at row {line_number}"
                    )
                seen.add(key)
                probability = float(payload["pair_probability"])
                cosine = float(payload["cosine_similarity"])
                left_rank = int(payload["left_rank"])
                right_rank = int(payload["right_rank"])
                if (
                    not math.isfinite(probability)
                    or not 0.0 <= probability <= 1.0
                    or not math.isfinite(cosine)
                    or not -1.0001 <= cosine <= 1.0001
                    or left_rank <= 0
                    or right_rank <= 0
                    or not isinstance(payload["variant_conflict"], bool)
                ):
                    raise DataValidationError(f"Invalid scored-pair values at row {line_number}")
                result.append(
                    ScoredPair(
                        left_posting_id=left_id,
                        right_posting_id=right_id,
                        left_index=index_by_id[left_id],
                        right_index=index_by_id[right_id],
                        cosine_similarity=cosine,
                        pair_probability=probability,
                        left_rank=left_rank,
                        right_rank=right_rank,
                        variant_conflict=payload["variant_conflict"],
                    )
                )
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
        raise DataValidationError("Cannot load the frozen scored-pair cache") from exc
    if len(result) != expected_count:
        raise DataValidationError("Frozen scored-pair count does not match entity metrics")
    return result


def _load_scored_pairs(
    config: EntityRecallRecoveryConfig, posting_ids: tuple[str, ...]
) -> list[ScoredPair]:
    return load_scored_pairs_file(
        config.source.scored_pairs_path,
        posting_ids,
        expected_count=int(config.source.metrics["pair_scoring"]["unique_pairs"]),
    )


def _policy_payload(policy: SingletonAttachmentPolicy) -> dict[str, object]:
    return {
        "enabled": policy.enabled,
        "probability_threshold": policy.probability_threshold,
        "reciprocal_rank": policy.reciprocal_rank,
        "minimum_support": policy.minimum_support,
        "target_margin": policy.target_margin,
    }


def _gate_payload(
    cluster: dict[str, Any], acceptance: RecoveryAcceptanceConfig
) -> tuple[bool, bool, dict[str, bool]]:
    pairwise = cluster["pairwise"]
    checks = {
        "minimum_pairwise_precision": (
            pairwise["precision"] >= acceptance.minimum_pairwise_precision
        ),
        "minimum_pairwise_recall": pairwise["recall"] >= acceptance.minimum_pairwise_recall,
        "minimum_pairwise_f1": pairwise["f1"] >= acceptance.minimum_pairwise_f1,
        "minimum_b_cubed_f1": cluster["b_cubed"]["f1"] >= acceptance.minimum_b_cubed_f1,
        "maximum_false_merge_pair_rate": (
            cluster["false_merge_pair_rate"] <= acceptance.maximum_false_merge_pair_rate
        ),
        "maximum_false_split_group_rate": (
            cluster["false_split_group_rate"] <= acceptance.maximum_false_split_group_rate
        ),
    }
    safety = checks["minimum_pairwise_precision"] and checks["maximum_false_merge_pair_rate"]
    return safety, all(checks.values()), checks


def _selection_key(trial: dict[str, Any]) -> tuple[float, ...]:
    cluster = trial["clustering"]
    pairwise = cluster["pairwise"]
    return (
        float(trial["passes_quality_target"]),
        float(trial["passes_safety_gate"]),
        float(cluster["b_cubed"]["f1"]),
        float(pairwise["f1"]),
        -float(cluster["false_split_group_rate"]),
        float(pairwise["precision"]),
        -float(trial["singleton_attachment"]["enabled"]),
    )


def _render_report(run: dict[str, Any]) -> str:
    selected = run["selection"]["selected"]
    incumbent = run["selection"]["incumbent"]
    cluster = selected["clustering"]
    pairwise = cluster["pairwise"]
    b3 = cluster["b_cubed"]
    policy = selected["singleton_attachment"]
    graph = selected["graph"]
    target_rows = "\n".join(
        f"| {name} | `{str(value).lower()}` |"
        for name, value in selected["acceptance_checks"].items()
    )
    false_merge_row = (
        f"| False-merge pair rate | {incumbent['false_merge_pair_rate']:.5f} | "
        f"{cluster['false_merge_pair_rate']:.5f} |"
    )
    false_split_row = (
        f"| False-split group rate | {incumbent['false_split_group_rate']:.5f} | "
        f"{cluster['false_split_group_rate']:.5f} |"
    )
    return f"""# Entity-Resolution Recall Recovery

Status: **{run["status"]}**. This experiment reuses frozen validation pair scores and does not
access test data or retrain a model.

## Result

| Metric | Incumbent | Selected recovery policy |
|---|---:|---:|
| Pairwise precision | {incumbent["pairwise"]["precision"]:.5f} | {pairwise["precision"]:.5f} |
| Pairwise recall | {incumbent["pairwise"]["recall"]:.5f} | {pairwise["recall"]:.5f} |
| Pairwise F1 | {incumbent["pairwise"]["f1"]:.5f} | {pairwise["f1"]:.5f} |
| B-cubed F1 | {incumbent["b_cubed"]["f1"]:.5f} | {b3["f1"]:.5f} |
{false_merge_row}
{false_split_row}

## Selected policy

- Core pair threshold / reciprocal rank / component coverage:
  `{selected["pair_probability_threshold"]:.3f}` / `{selected["reciprocal_rank"]}` /
  `{selected["cross_component_minimum_coverage"]:.2f}`
- Singleton attachment enabled: `{str(policy["enabled"]).lower()}`
- Singleton probability threshold / rank: `{policy["probability_threshold"]}` /
  `{policy["reciprocal_rank"]}`
- Independent supporting members: `{policy["minimum_support"]}`
- Ambiguous-target score margin: `{policy["target_margin"]}`
- Successful singleton attachments: `{graph["singleton_attachments"]}`

The second pass can attach an isolated listing only to a cluster that already contains at least
two listings and only when distinct members provide independent evidence. Membership is frozen
before the pass, so newly attached listings cannot bootstrap a transitive chain.

## Acceptance checks

| Check | Passed |
|---|---:|
{target_rows}

The selected policy remains validation-only. It must not replace the frozen final-system policy
or be evaluated on test until the acceptance protocol is explicitly frozen.

## Reproduction

```powershell
.venv\\Scripts\\shopee-entity-resolution recover-recall `
  --config configs\\experiment\\entity_recall_recovery.yaml
```
"""


def sweep_recovery_policies(
    config: EntityRecallRecoveryConfig,
    posting_ids: tuple[str, ...],
    pairs: list[ScoredPair],
    label_by_id: dict[str, str],
) -> tuple[list[dict[str, Any]], dict[str, Any], list[ClusterAssignment]]:
    """Evaluate the frozen label-blind graph grid and select by validation-only metrics."""
    trials: list[dict[str, Any]] = []
    selected_trial: dict[str, Any] | None = None
    selected_assignments: list[ClusterAssignment] | None = None
    total_trials = (
        len(config.selection.pair_probability_thresholds)
        * len(config.selection.reciprocal_rank_values)
        * len(config.selection.cross_component_coverage_values)
        * len(config.selection.singleton_attachment_policies)
    )
    completed = 0
    for threshold in config.selection.pair_probability_thresholds:
        for rank in config.selection.reciprocal_rank_values:
            for coverage in config.selection.cross_component_coverage_values:
                for policy in config.selection.singleton_attachment_policies:
                    assignments, diagnostics = build_conservative_clusters(
                        posting_ids,
                        pairs,
                        pair_probability_threshold=threshold,
                        reciprocal_rank=rank,
                        cross_component_minimum_coverage=coverage,
                        variant_conflict_override_probability=(
                            config.selection.variant_conflict_override_probability
                        ),
                        maximum_cluster_size=config.selection.maximum_cluster_size,
                        manual_review_margin=config.selection.manual_review_margin,
                        singleton_attachment=policy.enabled,
                        singleton_probability_threshold=policy.probability_threshold,
                        singleton_reciprocal_rank=policy.reciprocal_rank,
                        singleton_minimum_support=policy.minimum_support,
                        singleton_target_margin=policy.target_margin,
                    )
                    cluster = clustering_metrics(assignments, label_by_id)
                    safety, target, checks = _gate_payload(cluster, config.selection.acceptance)
                    trial: dict[str, Any] = {
                        "pair_probability_threshold": threshold,
                        "reciprocal_rank": rank,
                        "cross_component_minimum_coverage": coverage,
                        "singleton_attachment": _policy_payload(policy),
                        "passes_safety_gate": safety,
                        "passes_quality_target": target,
                        "acceptance_checks": checks,
                        "clustering": cluster,
                        "graph": asdict(diagnostics),
                    }
                    trials.append(trial)
                    if selected_trial is None or _selection_key(trial) > _selection_key(
                        selected_trial
                    ):
                        selected_trial = trial
                        selected_assignments = assignments
                    completed += 1
                    if completed % 20 == 0 or completed == total_trials:
                        LOGGER.info(
                            "Recall-recovery sweep: %d/%d policies", completed, total_trials
                        )
    if selected_trial is None or selected_assignments is None:
        raise DataValidationError("Recall-recovery selection grid produced no trials")
    return trials, selected_trial, selected_assignments


def run_entity_recall_recovery(config_path: Path) -> dict[str, object]:
    """Sweep conservative core and singleton policies on frozen validation pair scores."""
    config = load_entity_recall_recovery_config(config_path)
    existing = [
        str(path)
        for path in (
            config.artifacts.assignments,
            config.artifacts.metrics,
            config.artifacts.review,
            config.artifacts.report,
        )
        if path.exists()
    ]
    if existing:
        raise OutputConflictError(
            "Refusing to overwrite recall-recovery evidence: " + ", ".join(existing)
        )
    started = time.perf_counter()
    seed_everything(config.seed, deterministic=True)
    entity = config.source.experiment
    source_experiment = entity.source.experiment.source.experiment.source.experiment
    split = load_named_split(
        source_experiment.data.metadata_csv,
        source_experiment.data.split_manifest,
        "validation",
    )
    posting_ids = tuple(item.posting_id for item in split.items)
    pairs = _load_scored_pairs(config, posting_ids)
    LOGGER.info("Loaded %d frozen pair scores; starting validation policy sweep", len(pairs))

    trials, selected_trial, selected_assignments = sweep_recovery_policies(
        config, posting_ids, pairs, split.label_by_id
    )

    review = _failure_review(
        split,
        selected_assignments,
        example_limit=config.selection.failure_example_limit,
    )
    commit, dirty = _git_state()
    status = (
        "entity_recall_recovery_target_reached_validation_only"
        if selected_trial["passes_quality_target"]
        else "entity_recall_recovery_target_not_reached_validation_only"
    )
    incumbent_cluster = config.source.metrics["selection"]["selected"]["clustering"]
    run: dict[str, Any] = {
        "pipeline_version": "entity_resolution.recall_recovery.v1",
        "status": status,
        "provenance": {
            "config_sha256": canonical_text_sha256(config.config_path),
            "entity_config_sha256": config.source.entity_config_sha256,
            "entity_metrics_sha256": config.source.entity_metrics_sha256,
            "scored_pairs_sha256": config.source.scored_pairs_sha256,
            "split_manifest_sha256": sha256_file(source_experiment.data.split_manifest),
            "git_commit": commit,
            "git_dirty": dirty,
            "seed": config.seed,
            "python": platform.python_version(),
            "numpy": np.__version__,
        },
        "data": {"split": "validation", "listings": len(posting_ids), "test_accessed": False},
        "source": {
            "unique_pairs": len(pairs),
            "pair_scores_recomputed": False,
            "model_retrained": False,
        },
        "selection": {
            "objective": "maximize_clustering_quality_subject_to_precision_and_false_merge_safety",
            "acceptance": asdict(config.selection.acceptance),
            "incumbent": incumbent_cluster,
            "selected": selected_trial,
            "trials": trials,
        },
        "group_size_strata": group_size_strata(selected_assignments, split.label_by_id),
        "failure_analysis": review["counts"],
        "runtime_seconds": time.perf_counter() - started,
        "artifacts": {
            "assignments": str(config.artifacts.assignments),
            "review": str(config.artifacts.review),
            "report": str(config.artifacts.report),
        },
        "test": {"status": "disabled_recall_recovery_validation_only"},
    }
    _write_assignments_atomic(config.artifacts.assignments, selected_assignments)
    _write_text_atomic(config.artifacts.review, json.dumps(review, indent=2, sort_keys=True) + "\n")
    _write_text_atomic(config.artifacts.metrics, json.dumps(run, indent=2, sort_keys=True) + "\n")
    _write_text_atomic(config.artifacts.report, _render_report(run))
    cluster = cast(dict[str, Any], selected_trial["clustering"])
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
    }
