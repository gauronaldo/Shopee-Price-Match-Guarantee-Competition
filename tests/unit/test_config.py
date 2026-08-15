from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from shopee_match.config import load_config
from shopee_match.errors import ConfigurationError

SMOKE_CONFIG = Path("configs/smoke.yaml")


def test_load_smoke_config() -> None:
    config = load_config(SMOKE_CONFIG)

    assert config.config_version == "phase0.smoke.v1"
    assert config.project.seed == 2026
    assert config.data.metadata_csv == Path("tests/fixtures/smoke/train.csv")
    assert config.output.review_threshold < config.output.match_threshold


def test_config_rejects_ambiguous_threshold_order(tmp_path: Path) -> None:
    raw = yaml.safe_load(SMOKE_CONFIG.read_text(encoding="utf-8"))
    raw["output"]["review_threshold"] = raw["output"]["match_threshold"]
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    with pytest.raises(ConfigurationError, match="Thresholds"):
        load_config(path)


@pytest.mark.parametrize("unsafe_path", ["../private/train.csv", "C:\\Users\\name\\train.csv"])
def test_config_rejects_machine_or_parent_paths(tmp_path: Path, unsafe_path: str) -> None:
    raw = yaml.safe_load(SMOKE_CONFIG.read_text(encoding="utf-8"))
    raw["data"]["metadata_csv"] = unsafe_path
    path = tmp_path / "bad-path.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    with pytest.raises(ConfigurationError, match="project-relative"):
        load_config(path)
