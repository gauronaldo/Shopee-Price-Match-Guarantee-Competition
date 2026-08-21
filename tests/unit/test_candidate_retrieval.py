"""Exact and approximate Phase 7 vector-index contracts."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from shopee_match.errors import ConfigurationError
from shopee_match.retrieval import (
    ExactCosineIndex,
    FaissHnswIndex,
    normalize_embeddings,
    search_result_to_ranking,
)
from shopee_match.retrieval.config import load_candidate_retrieval_config


def _fixture() -> tuple[tuple[str, ...], np.ndarray]:
    posting_ids = ("a1", "a2", "b1", "b2", "c1", "c2")
    embeddings = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.99, 0.01, 0.0],
            [0.0, 1.0, 0.0],
            [0.01, 0.99, 0.0],
            [0.0, 0.0, 1.0],
            [0.01, 0.0, 0.99],
        ],
        dtype=np.float32,
    )
    return posting_ids, embeddings


def test_exact_index_excludes_self_and_round_trips(tmp_path: Path) -> None:
    posting_ids, embeddings = _fixture()
    index = ExactCosineIndex(posting_ids, embeddings)
    indices, scores = index.search(embeddings, 2, query_ids=posting_ids, block_size=2)
    ranking = search_result_to_ranking(posting_ids, posting_ids, indices, scores)
    assert ranking["a1"][0].posting_id == "a2"
    assert ranking["b1"][0].posting_id == "b2"
    assert ranking["c1"][0].posting_id == "c2"
    assert all(query not in {row.posting_id for row in ranking[query]} for query in posting_ids)

    path = tmp_path / "exact.npz"
    index.save(path)
    restored = ExactCosineIndex.load(path)
    np.testing.assert_array_equal(restored.vectors, index.vectors)
    restored_indices, restored_scores = restored.search(
        embeddings, 2, query_ids=posting_ids, block_size=3
    )
    assert np.array_equal(indices, restored_indices)
    assert np.allclose(scores, restored_scores)


def test_faiss_hnsw_agrees_with_exact_and_round_trips(tmp_path: Path) -> None:
    posting_ids, embeddings = _fixture()
    exact = ExactCosineIndex(posting_ids, embeddings)
    expected_indices, _expected_scores = exact.search(embeddings, 2, query_ids=posting_ids)
    approximate = FaissHnswIndex(
        posting_ids,
        embeddings,
        m=4,
        ef_construction=40,
        ef_search=32,
        threads=1,
    )
    actual_indices, actual_scores = approximate.search(embeddings, 2, query_ids=posting_ids)
    assert np.array_equal(actual_indices, expected_indices)

    index_path = tmp_path / "hnsw.faiss"
    metadata_path = tmp_path / "hnsw.json"
    approximate.save(index_path, metadata_path)
    restored = FaissHnswIndex.load(index_path, metadata_path)
    loaded_indices, loaded_scores = restored.search(embeddings, 2, query_ids=posting_ids)
    assert np.array_equal(loaded_indices, actual_indices)
    assert np.allclose(loaded_scores, actual_scores)


def test_normalization_rejects_zero_and_non_finite_vectors() -> None:
    with pytest.raises(ValueError, match="non-zero"):
        normalize_embeddings(np.zeros((2, 3), dtype=np.float32))
    invalid = np.ones((2, 3), dtype=np.float32)
    invalid[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        normalize_embeddings(invalid)


def test_candidate_retrieval_config_rejects_test_selection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import shopee_match.retrieval.config as config_module

    metrics_path = tmp_path / "metrics.json"
    metrics_path.write_text(
        json.dumps(
            {
                "pipeline_version": "phase6.hard_negative_training.v1",
                "acceptance": {"pilot_pass": True},
                "test": {"status": "disabled_phase6_validation_only"},
                "data": {"test_accessed": False},
            }
        ),
        encoding="utf-8",
    )
    paths = {
        "phase6_config": tmp_path / "phase6.yaml",
        "checkpoint": tmp_path / "best.pt",
        "metrics": metrics_path,
        "mined_manifest": tmp_path / "pairs.jsonl",
    }
    experiment = SimpleNamespace(
        seed=2026,
        artifacts=SimpleNamespace(
            checkpoint=paths["checkpoint"],
            metrics=paths["metrics"],
            manifest=paths["mined_manifest"],
        ),
    )
    monkeypatch.setattr(
        config_module,
        "_verified_file",
        lambda _raw, name, **_kwargs: (paths[name], "0" * 64),
    )
    monkeypatch.setattr(
        config_module,
        "load_hard_negative_experiment_config",
        lambda _path: experiment,
    )
    config_path = tmp_path / "candidate.yaml"
    config_path.write_text(
        """config_version: phase7.candidate_retrieval.v1
seed: 2026
source:
  phase6_config: configs/phase6.yaml
  phase6_config_sha256: 0000000000000000000000000000000000000000000000000000000000000000
  checkpoint: artifacts/best.pt
  checkpoint_sha256: 0000000000000000000000000000000000000000000000000000000000000000
  metrics: artifacts/metrics.json
  metrics_sha256: 0000000000000000000000000000000000000000000000000000000000000000
  mined_manifest: artifacts/pairs.jsonl
  mined_manifest_sha256: 0000000000000000000000000000000000000000000000000000000000000000
data:
  split: validation
  evaluate_test: false
embedding: {device: cpu, batch_size: 8, num_workers: 0}
exact: {block_size: 8}
faiss:
  index_type: hnsw_flat
  metric: inner_product
  m: 4
  ef_construction: 20
  ef_search_values: [8, 16]
  threads: 1
  rerank_buffer: 2
selection:
  k_values: [1, 2]
  target_recall: 0.9
  maximum_approximate_recall_drop: 0.01
  minimum_exact_candidate_agreement: 0.9
  latency_query_count: 2
  latency_repetitions: 1
artifacts:
  root: artifacts/retrieval
  embedding_cache: artifacts/retrieval/embeddings.npz
  exact_index: artifacts/retrieval/exact.npz
  faiss_index: artifacts/retrieval/hnsw.faiss
  faiss_metadata: artifacts/retrieval/hnsw.json
  metrics: artifacts/retrieval/metrics.json
  review: artifacts/retrieval/review.json
  report: reports/retrieval.md
""",
        encoding="utf-8",
    )
    assert load_candidate_retrieval_config(config_path).selection.k_values == (1, 2)
    invalid = config_path.read_text(encoding="utf-8").replace(
        "evaluate_test: false", "evaluate_test: true"
    )
    config_path.write_text(invalid, encoding="utf-8")
    with pytest.raises(ConfigurationError, match="validation only"):
        load_candidate_retrieval_config(config_path)
