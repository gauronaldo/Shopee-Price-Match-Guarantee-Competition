from __future__ import annotations

from pathlib import Path

from shopee_match.data.pair_recall_protocol import (
    PairRecallProtocolConfig,
    create_pair_recall_protocol,
)


def test_pair_recall_protocol_is_scoped_and_group_disjoint(tmp_path: Path) -> None:
    rows = []
    for index in range(4):
        rows.append(
            {
                "posting_id": f"train_{index}",
                "label_group": f"train_group_{index // 2}",
                "split": "train",
                "super_component_id": f"train_component_{index // 2}",
            }
        )
    for index in range(8):
        rows.append(
            {
                "posting_id": f"development_{index}",
                "label_group": f"development_group_{index // 2}",
                "split": "development",
                "super_component_id": f"development_component_{index // 2}",
            }
        )
    rows.extend(
        [
            {
                "posting_id": "confirmation_0",
                "label_group": "confirmation_group",
                "split": "confirmation",
                "super_component_id": "confirmation_component",
            },
            {
                "posting_id": "historical_0",
                "label_group": "historical_group",
                "split": "historical_test",
                "super_component_id": "historical_component",
            },
        ]
    )
    config = PairRecallProtocolConfig(
        tmp_path / "source.jsonl",
        "a" * 64,
        "train",
        "development",
        17,
        0.5,
        tmp_path / "modeling.jsonl",
        tmp_path / "summary.json",
        tmp_path / "config.yaml",
    )
    config.config_path.write_text("fixture", encoding="utf-8")
    output, summary = create_pair_recall_protocol(rows, config)

    assert summary["integrity"] == {
        "label_groups_cross_split": 0,
        "super_components_cross_split": 0,
        "v2_confirmation_rows_exposed": 0,
    }
    assert summary["excluded_source_roles"] == {"confirmation": 1, "historical_test": 1}
    assert {row["split"] for row in output} == {"train", "validation", "test"}
    assert all(not row["posting_id"].startswith("confirmation") for row in output)
    assert all(not row["posting_id"].startswith("historical") for row in output)
