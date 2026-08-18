from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from shopee_match.data.config import load_phase1_config
from shopee_match.errors import ConfigurationError

from ..dataset_helpers import make_dataset_workspace


def test_dataset_config_rejects_non_label_group_split(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_root = Path.cwd()
    config_path = make_dataset_workspace(tmp_path, source_root)
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    raw["split"]["group_key"] = "posting_id"
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    with pytest.raises(ConfigurationError, match="label_group"):
        load_phase1_config(Path("phase1.yaml"))
