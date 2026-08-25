from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from shopee_match.errors import ConfigurationError
from shopee_match.evaluation import hybrid_system_config as module
from shopee_match.evaluation.hybrid_system_config import HybridSystemArtifacts
from shopee_match.evaluation.hybrid_system_evaluator import _existing_outputs


def _config_text(pair_threshold: float) -> str:
    return f"""config_version: hybrid.system_evaluation.v1
seed: 2026
frozen:
  entity_config: configs/experiment/hybrid_entity_resolution.yaml
  entity_config_sha256: {"a" * 64}
  entity_metrics: artifacts/entity_resolution/hybrid_validation/metrics.json
  entity_metrics_sha256: {"b" * 64}
  policy:
    candidate_k: 75
    pair_probability_threshold: {pair_threshold}
    reciprocal_rank: 5
    cross_component_minimum_coverage: 1.0
    variant_conflict_override_probability: 0.15
    maximum_cluster_size: 64
    manual_review_margin: 0.02
    singleton_attachment:
      enabled: true
      probability_threshold: 0.18
      reciprocal_rank: 50
      minimum_support: 2
      target_margin: 0.02
data:
  split: test
  evaluation_manifest: data/splits/shopee_confirmation_modeling.jsonl
  evaluation_manifest_sha256: {"c" * 64}
  evaluate_once: true
  allow_test_selection: false
  attempt_number: 3
  prior_failed_attempts: 2
runtime:
  device: cpu
  embedding_batch_size: 8
  pair_batch_size: 16
  num_workers: 0
evaluation:
  metric_k_values: [20, 50, 75]
  exact_block_size: 8
  latency_query_count: 4
  latency_repetitions: 1
  calibration_bins: 5
  required_recall: 0.8
  required_precision: 0.9
  failure_example_limit: 3
artifacts:
  root: artifacts/final_evaluation/hybrid_fixture
  access_marker: artifacts/final_evaluation/hybrid_fixture/test_access_started.json
  embeddings: artifacts/final_evaluation/hybrid_fixture/embeddings.npz
  ranking: artifacts/final_evaluation/hybrid_fixture/ranking.jsonl
  scored_pairs: artifacts/final_evaluation/hybrid_fixture/scored_pairs.jsonl
  assignments: artifacts/final_evaluation/hybrid_fixture/entity_assignments.csv
  metrics: artifacts/final_evaluation/hybrid_fixture/metrics.json
  review: artifacts/final_evaluation/hybrid_fixture/failure_review.json
  report: artifacts/final_evaluation/hybrid_fixture/report.md
"""


def _patch_sources(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    entity_path = tmp_path / "hybrid_entity.yaml"
    metrics_path = tmp_path / "metrics.json"
    selected = {
        "passes_quality_target": True,
        "pair_probability_threshold": 0.14,
        "reciprocal_rank": 5,
        "cross_component_minimum_coverage": 1.0,
        "singleton_attachment": {
            "enabled": True,
            "probability_threshold": 0.18,
            "reciprocal_rank": 50,
            "minimum_support": 2,
            "target_margin": 0.02,
        },
    }
    metrics = {
        "pipeline_version": "hybrid_entity_resolution.v1",
        "status": "hybrid_entity_target_reached_validation_only",
        "data": {"split": "validation", "test_accessed": False},
        "provenance": {"config_sha256": "a" * 64, "git_dirty": False},
        "source": {"candidate_k": 75},
        "selection": {"selected": selected},
    }
    metrics_path.write_text(json.dumps(metrics), encoding="utf-8")
    entity_path.write_text("fixture", encoding="utf-8")
    selection = SimpleNamespace(
        variant_conflict_override_probability=0.15,
        maximum_cluster_size=64,
        manual_review_margin=0.02,
    )
    entity_config: Any = SimpleNamespace(
        seed=2026,
        artifacts=SimpleNamespace(metrics=metrics_path),
        source=SimpleNamespace(
            hybrid_metrics={
                "provenance": {"git_dirty": False},
                "hybrid": {
                    "retrieval_curve": {"20": {}, "50": {}, "75": {}}
                },
            },
            recovery=SimpleNamespace(selection=selection),
            hybrid=SimpleNamespace(
                source=SimpleNamespace(
                    experiment=SimpleNamespace(
                        source=SimpleNamespace(
                            experiment=SimpleNamespace(
                                source=SimpleNamespace(
                                    experiment=SimpleNamespace(
                                        data=SimpleNamespace(
                                            split_manifest=tmp_path / "training.jsonl"
                                        )
                                    )
                                )
                            )
                        )
                    )
                )
            ),
        ),
    )

    def verified(
        _raw: dict[str, Any], name: str, *, portable_text: bool = False
    ) -> tuple[Path, str]:
        del portable_text
        return (entity_path, "a" * 64) if name == "entity_config" else (metrics_path, "b" * 64)

    monkeypatch.setattr(module, "_verified_file", verified)
    monkeypatch.setattr(module, "load_hybrid_entity_config", lambda _path: entity_config)
    monkeypatch.setattr(module, "sha256_file", lambda _path: "c" * 64)


def test_hybrid_final_config_accepts_validation_frozen_policy(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _patch_sources(monkeypatch, tmp_path)
    config_path = tmp_path / "hybrid_final.yaml"
    config_path.write_text(_config_text(0.14), encoding="utf-8")
    config = module.load_hybrid_system_evaluation_config(config_path)
    assert config.policy.candidate_k == 75
    assert config.policy.singleton_attachment.minimum_support == 2
    assert config.attempt_number == 3
    assert config.prior_failed_attempts == 2


def test_hybrid_final_config_rejects_threshold_drift(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _patch_sources(monkeypatch, tmp_path)
    config_path = tmp_path / "hybrid_final.yaml"
    config_path.write_text(_config_text(0.15), encoding="utf-8")
    with pytest.raises(ConfigurationError, match="differs from validation-selected"):
        module.load_hybrid_system_evaluation_config(config_path)


def test_hybrid_final_config_rejects_k_missing_from_validation_curve(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _patch_sources(monkeypatch, tmp_path)
    original_loader = module.load_hybrid_entity_config
    entity_config = original_loader(Path("unused"))
    entity_config.source.hybrid_metrics["hybrid"]["retrieval_curve"].pop("20")
    config_path = tmp_path / "hybrid_final.yaml"
    config_path.write_text(_config_text(0.14), encoding="utf-8")
    with pytest.raises(ConfigurationError, match="frozen validation retrieval curve"):
        module.load_hybrid_system_evaluation_config(config_path)


def test_hybrid_final_output_guard_detects_access_marker(tmp_path: Path) -> None:
    artifacts = HybridSystemArtifacts(
        root=tmp_path,
        access_marker=tmp_path / "access.json",
        embeddings=tmp_path / "embeddings.npz",
        ranking=tmp_path / "ranking.jsonl",
        scored_pairs=tmp_path / "pairs.jsonl",
        assignments=tmp_path / "assignments.csv",
        metrics=tmp_path / "metrics.json",
        review=tmp_path / "review.json",
        report=tmp_path / "report.md",
    )
    assert _existing_outputs(artifacts) == []
    artifacts.access_marker.write_text("{}", encoding="utf-8")
    assert _existing_outputs(artifacts) == [str(artifacts.access_marker)]
