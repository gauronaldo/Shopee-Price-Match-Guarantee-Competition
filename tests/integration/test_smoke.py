from __future__ import annotations

from pathlib import Path

from shopee_match.config import load_config
from shopee_match.smoke import main, validate_smoke_fixture


def test_fixture_validation_is_deterministic() -> None:
    config_path = Path("configs/smoke.yaml")
    config = load_config(config_path)

    first = validate_smoke_fixture(config, Path.cwd(), config_path)
    second = validate_smoke_fixture(config, Path.cwd(), config_path)

    assert first == second
    assert first.status == "ok"
    assert first.records == 6
    assert first.label_groups == 3
    assert first.group_size_distribution == {2: 3}


def test_smoke_cli_succeeds() -> None:
    assert main(["--config", "configs/smoke.yaml"]) == 0
