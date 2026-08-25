from __future__ import annotations

from shopee_match.evaluation.catalog_attachment_protocol import (
    assign_catalog_roles,
    validate_catalog_roles,
)
from shopee_match.evaluation.protocol import CorpusItem, EvaluationSplit


def _split() -> EvaluationSplit:
    labels = {
        "a1": "a",
        "a2": "a",
        "a3": "a",
        "b1": "b",
        "b2": "b",
        "c1": "c",
    }
    items = tuple(
        CorpusItem(posting_id, f"{posting_id}.jpg", posting_id.ljust(16, "0"), posting_id)
        for posting_id in labels
    )
    return EvaluationSplit(items, labels)


def test_catalog_roles_are_deterministic_and_label_safe() -> None:
    split = _split()
    first = assign_catalog_roles(
        split,
        seed=2027,
        partition="development",
        known_entity_fraction=0.5,
    )
    second = assign_catalog_roles(
        split,
        seed=2027,
        partition="development",
        known_entity_fraction=0.5,
    )
    assert first == second
    validate_catalog_roles(split, first)
    role_by_id = {row.posting_id: row.role for row in first}
    assert role_by_id["c1"] == "new_entity_query"
    assert sum(role == "catalog_reference" for role in role_by_id.values()) == 1
    assert sum(role == "known_entity_query" for role in role_by_id.values()) >= 1


def test_role_manifest_never_contains_label_group() -> None:
    roles = assign_catalog_roles(
        _split(),
        seed=2027,
        partition="development",
        known_entity_fraction=0.5,
    )
    assert all(not hasattr(row, "label_group") for row in roles)
