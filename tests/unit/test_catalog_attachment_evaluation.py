from __future__ import annotations

from pathlib import Path

from shopee_match.evaluation.catalog_attachment_config import (
    CatalogAttachmentComparison,
    CatalogAttachmentSafety,
)
from shopee_match.evaluation.catalog_attachment_evaluator import (
    PairEvidence,
    attachment_metrics,
    cross_phash_ranking,
    passes_comparison,
    passes_safety,
)
from shopee_match.evaluation.catalog_attachment_protocol import CatalogRole
from shopee_match.evaluation.protocol import CorpusItem


def _item(identifier: str, phash: str) -> CorpusItem:
    return CorpusItem(identifier, f"{identifier}.jpg", phash, identifier)


def test_cross_phash_ranking_uses_only_catalog_references() -> None:
    references = (_item("r1", "0000000000000000"), _item("r2", "ffffffffffffffff"))
    queries = (_item("q1", "0000000000000001"),)
    ranking = cross_phash_ranking(queries, references, 2)
    assert [row.posting_id for row in ranking["q1"]] == ["r1", "r2"]
    assert "q1" not in {row.posting_id for row in ranking["q1"]}


def test_attachment_metrics_separate_known_and_new_queries() -> None:
    roles = (
        CatalogRole("r1", "development", "catalog_reference"),
        CatalogRole("q1", "development", "known_entity_query"),
        CatalogRole("q2", "development", "new_entity_query"),
    )
    labels = {"r1": "a", "q1": "a", "q2": "b"}
    evidence: dict[str, tuple[PairEvidence, ...]] = {
        "q1": (PairEvidence("q1", "r1", 1, 0.9, False),),
        "q2": (PairEvidence("q2", "r1", 1, 0.1, False),),
    }
    metrics = attachment_metrics(
        evidence,
        roles,
        labels,
        threshold=0.5,
        manual_review_margin=0.0,
        target_margin=0.0,
        variant_conflict_override_probability=0.8,
    )
    assert metrics["attachment_precision"] == 1.0
    assert metrics["attachment_recall"] == 1.0
    assert metrics["new_entity_detection_recall"] == 1.0
    assert metrics["new_entity_false_attachment_rate"] == 0.0


def test_safety_gate_requires_both_match_and_new_entity_quality() -> None:
    safety = CatalogAttachmentSafety(0.88, 0.8, 0.12, 0.12, 0.3)
    metrics = {
        "attachment_precision": 0.9,
        "new_entity_detection_recall": 0.85,
        "new_entity_false_attachment_rate": 0.1,
        "overall_false_attachment_rate": 0.05,
        "manual_review_rate": 0.2,
    }
    assert passes_safety(metrics, safety)
    metrics["new_entity_false_attachment_rate"] = 0.2
    assert not passes_safety(metrics, safety)


def test_candidate_comparison_requires_material_safe_improvement() -> None:
    baseline = {
        "selection": {
            "selected": {
                "attachment_precision": 0.92,
                "attachment_recall": 0.75,
                "attachment_f1": 0.82,
                "new_entity_detection_recall": 0.88,
                "new_entity_false_attachment_rate": 0.09,
                "overall_false_attachment_rate": 0.05,
                "manual_review_rate": 0.05,
            }
        }
    }
    comparison = CatalogAttachmentComparison(
        Path("baseline.json"), baseline, 0.02, 0.01, 0.01, 0.01, 0.01, 0.01, 0.05
    )
    candidate = {
        "attachment_precision": 0.915,
        "attachment_recall": 0.78,
        "attachment_f1": 0.84,
        "new_entity_detection_recall": 0.875,
        "new_entity_false_attachment_rate": 0.095,
        "overall_false_attachment_rate": 0.055,
        "manual_review_rate": 0.06,
    }
    assert passes_comparison(candidate, comparison)
    candidate["attachment_precision"] = 0.89
    assert not passes_comparison(candidate, comparison)
