from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from shopee_match.errors import ConfigurationError
from shopee_match.evaluation import final_system_config as module


def _config_text(pair_threshold: float) -> str:
    return f"""config_version: final.system_evaluation.v1
seed: 2026
frozen:
  entity_config: configs/experiment/entity_resolution_benchmark.yaml
  entity_config_sha256: {"a" * 64}
  entity_metrics: artifacts/entity_resolution/validation/metrics.json
  entity_metrics_sha256: {"b" * 64}
  candidate_k: 50
  pair_probability_threshold: {pair_threshold}
  reciprocal_rank: 5
  cross_component_minimum_coverage: 1.0
  variant_conflict_override_probability: 0.15
  maximum_cluster_size: 64
  manual_review_margin: 0.02
data:
  split: test
  evaluate_once: true
  allow_test_selection: false
runtime:
  device: cpu
  embedding_batch_size: 8
  pair_batch_size: 16
  num_workers: 0
evaluation:
  recall_at: [1, 20, 50]
  average_precision_at: [20, 50]
  exact_block_size: 8
  latency_query_count: 4
  latency_repetitions: 1
  calibration_bins: 5
  required_recall: 0.8
  required_precision: 0.9
  failure_example_limit: 3
artifacts:
  root: artifacts/final_evaluation/test_fixture
  access_marker: artifacts/final_evaluation/test_fixture/test_access_started.json
  embeddings: artifacts/final_evaluation/test_fixture/embeddings.npz
  scored_pairs: artifacts/final_evaluation/test_fixture/scored_pairs.jsonl
  assignments: artifacts/final_evaluation/test_fixture/entity_assignments.csv
  metrics: artifacts/final_evaluation/test_fixture/metrics.json
  review: artifacts/final_evaluation/test_fixture/failure_review.json
  report: reports/final_evaluation_fixture.md
"""


def _patch_frozen_sources(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    entity_path = tmp_path / "entity.yaml"
    metrics_path = tmp_path / "metrics.json"
    metrics = {
        "pipeline_version": "phase8.entity_resolution.v1",
        "status": "phase8_complete_validation_only",
        "data": {"split": "validation", "test_accessed": False},
        "provenance": {"config_sha256": "a" * 64, "git_dirty": False},
        "source": {"candidate_k": 50},
        "selection": {
            "variant_conflict_override_probability": 0.15,
            "maximum_cluster_size": 64,
            "selected": {
                "passes_precision_gate": True,
                "pair_probability_threshold": 0.16,
                "reciprocal_rank": 5,
                "cross_component_minimum_coverage": 1.0,
            },
        },
    }
    metrics_path.write_text(json.dumps(metrics), encoding="utf-8")
    entity_path.write_text("fixture", encoding="utf-8")
    entity_config: Any = SimpleNamespace(
        seed=2026,
        artifacts=SimpleNamespace(metrics=metrics_path),
        selection=SimpleNamespace(manual_review_margin=0.02),
    )

    def verified(
        _raw: dict[str, Any], name: str, *, portable_text: bool = False
    ) -> tuple[Path, str]:
        del portable_text
        return (
            (entity_path, "a" * 64)
            if name == "entity_config"
            else (
                metrics_path,
                "b" * 64,
            )
        )

    monkeypatch.setattr(module, "_verified_file", verified)
    monkeypatch.setattr(module, "load_entity_resolution_config", lambda _path: entity_config)


def test_final_config_accepts_exact_validation_frozen_policy(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _patch_frozen_sources(monkeypatch, tmp_path)
    config_path = tmp_path / "final.yaml"
    config_path.write_text(_config_text(0.16), encoding="utf-8")
    config = module.load_final_system_evaluation_config(config_path)
    assert config.policy.candidate_k == 50
    assert config.policy.pair_probability_threshold == pytest.approx(0.16)
    assert config.policy.cross_component_minimum_coverage == pytest.approx(1.0)


def test_final_config_rejects_pair_threshold_drift(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _patch_frozen_sources(monkeypatch, tmp_path)
    config_path = tmp_path / "final.yaml"
    config_path.write_text(_config_text(0.17), encoding="utf-8")
    with pytest.raises(ConfigurationError, match="differs from validation-selected"):
        module.load_final_system_evaluation_config(config_path)
