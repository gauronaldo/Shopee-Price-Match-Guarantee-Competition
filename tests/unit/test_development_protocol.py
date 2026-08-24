from __future__ import annotations

from pathlib import Path

from shopee_match.data.development_config import (
    DevelopmentAllocationConfig,
    DevelopmentArtifactConfig,
    DevelopmentProtocolConfig,
    DevelopmentSourceConfig,
)
from shopee_match.data.development_split import (
    SourceManifestRow,
    create_development_protocol,
    modeling_manifest,
)


def _config(tmp_path: Path) -> DevelopmentProtocolConfig:
    config_path = tmp_path / "protocol.yaml"
    config_path.write_text("config_version: test\n", encoding="utf-8")
    return DevelopmentProtocolConfig(
        source=DevelopmentSourceConfig(Path("source.jsonl"), "0" * 64, "test"),
        allocation=DevelopmentAllocationConfig(37, 0.8, 0.1, 0.1),
        artifacts=DevelopmentArtifactConfig(
            tmp_path / "development.jsonl",
            tmp_path / "development.modeling.jsonl",
            tmp_path / "development.summary.json",
        ),
        config_path=config_path,
    )


def _rows() -> tuple[SourceManifestRow, ...]:
    rows = []
    for group_index in range(40):
        source_split = "test" if group_index >= 36 else "train"
        for item_index in range(2 + group_index % 3):
            rows.append(
                SourceManifestRow(
                    posting_id=f"p{group_index:02d}_{item_index}",
                    image=f"p{group_index:02d}_{item_index}.jpg",
                    label_group=f"g{group_index:02d}",
                    source_split=source_split,
                    super_component_id=f"c{group_index:02d}",
                )
            )
    return tuple(rows)


def test_development_protocol_is_deterministic_and_group_disjoint(tmp_path: Path) -> None:
    config = _config(tmp_path)

    first, first_summary = create_development_protocol(_rows(), config)
    second, second_summary = create_development_protocol(_rows(), config)

    assert first == second
    assert first_summary == second_summary
    assert set(first_summary["listings"]) == {
        "train",
        "development",
        "confirmation",
        "historical_test",
    }
    assert not any(first_summary["integrity"].values())


def test_historical_test_membership_is_preserved_exactly(tmp_path: Path) -> None:
    rows = _rows()

    manifest, summary = create_development_protocol(rows, _config(tmp_path))

    expected = {row.posting_id for row in rows if row.source_split == "test"}
    actual = {row["posting_id"] for row in manifest if row["split"] == "historical_test"}
    assert actual == expected
    assert summary["historical_test_preserved"] is True
    assert summary["historical_test_used_for_selection"] is False


def test_super_component_is_never_split_between_protocol_roles(tmp_path: Path) -> None:
    rows = list(_rows())
    rows.append(
        SourceManifestRow(
            posting_id="p_extra",
            image="p_extra.jpg",
            label_group="g_extra",
            source_split="validation",
            super_component_id="c00",
        )
    )

    manifest, _summary = create_development_protocol(tuple(rows), _config(tmp_path))

    roles = {row["split"] for row in manifest if row["super_component_id"] == "c00"}
    assert len(roles) == 1


def test_modeling_manifest_exposes_development_as_validation_only(tmp_path: Path) -> None:
    manifest, _summary = create_development_protocol(_rows(), _config(tmp_path))

    compatibility = modeling_manifest(manifest)
    role_by_id = {row["posting_id"]: row["split"] for row in compatibility}

    assert set(role_by_id.values()) == {"train", "validation", "test"}
    for row in manifest:
        expected = {
            "train": "train",
            "development": "validation",
            "confirmation": "test",
            "historical_test": "test",
        }[row["split"]]
        assert role_by_id[row["posting_id"]] == expected
