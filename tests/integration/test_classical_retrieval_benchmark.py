from __future__ import annotations

import json
from pathlib import Path

import pytest

from shopee_match.evaluation.pipeline import run_classical_retrieval_benchmark

from ..benchmark_helpers import make_benchmark_workspace


def test_classical_retrieval_benchmark_runs_from_fixture_to_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_root = Path.cwd()
    make_benchmark_workspace(tmp_path, source_root)
    monkeypatch.chdir(tmp_path)

    result = run_classical_retrieval_benchmark(Path("classical_retrieval.yaml"))
    metrics = json.loads(Path(result["metrics"]).read_text(encoding="utf-8"))

    assert result["status"] == "complete"
    assert set(metrics["baselines"]) == {
        "phash",
        "tfidf",
        "orb",
        "fusion",
        "pair_matcher",
    }
    assert metrics["data"] == {"test": 2, "train": 4, "validation": 2}
    assert set(metrics["model"]["pair_matcher"]["standardized_coefficients"]) == {
        "tfidf_similarity",
        "phash_similarity",
        "orb_similarity",
        "token_jaccard",
        "exact_normalized_title",
        "digit_jaccard",
        "digit_conflict",
        "quantity_overlap",
        "quantity_conflict",
        "model_token_jaccard",
        "title_length_ratio",
    }
    assert metrics["analysis"]["candidate_ceiling"]["test"]["macro_recall"] == 1.0
    assert metrics["efficiency"]["process_peak_working_set_bytes"] > 0
    assert (
        metrics["baselines"]["fusion"]["runtime_seconds"]
        >= metrics["baselines"]["fusion"]["stage_runtime_seconds"]
    )
    assert Path(result["report"]).exists()
    assert Path(result["threshold_figure"]).read_text(encoding="utf-8").startswith("<svg")
    assert Path("artifacts/classical_retrieval/review_examples.json").exists()
    assert Path("artifacts/classical_retrieval/pair_matcher_failure_examples.json").exists()
