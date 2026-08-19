from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from shopee_match.errors import ConfigurationError
from shopee_match.evaluation.config import load_classical_retrieval_config

from ..benchmark_helpers import make_benchmark_workspace


def test_classical_retrieval_config_loads_strict_protocol(tmp_path: Path) -> None:
    config_path = make_benchmark_workspace(tmp_path, Path.cwd())

    config = load_classical_retrieval_config(config_path)

    assert config.evaluation.tune_split == "validation"
    assert config.evaluation.final_split == "test"
    assert config.tfidf.ngram_range == (2, 3)
    assert config.fusion.weight_grid == (0.0, 0.5, 1.0)
    assert config.pair_matcher.training_queries == 4
    assert config.pair_matcher.candidate_k_per_source == 1


def test_classical_retrieval_config_rejects_test_tuning(tmp_path: Path) -> None:
    config_path = make_benchmark_workspace(tmp_path, Path.cwd())
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    raw["evaluation"]["tune_split"] = "test"
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    with pytest.raises(ConfigurationError, match="tune on validation"):
        load_classical_retrieval_config(config_path)


def test_classical_retrieval_config_requires_rankings_to_cover_metrics(tmp_path: Path) -> None:
    config_path = make_benchmark_workspace(tmp_path, Path.cwd())
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    raw["evaluation"]["recall_at"] = [1, 5]
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    with pytest.raises(ConfigurationError, match="cover the largest"):
        load_classical_retrieval_config(config_path)
