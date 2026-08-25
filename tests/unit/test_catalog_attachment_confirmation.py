from __future__ import annotations

from shopee_match.evaluation.catalog_attachment_confirmation_config import ConfirmationStability
from shopee_match.evaluation.catalog_attachment_confirmation_evaluator import passes_stability


def _development() -> dict[str, float]:
    return {
        "attachment_precision": 0.92,
        "attachment_recall": 0.75,
        "attachment_f1": 0.83,
        "new_entity_detection_recall": 0.88,
        "new_entity_false_attachment_rate": 0.09,
        "overall_false_attachment_rate": 0.05,
        "manual_review_rate": 0.05,
    }


def test_confirmation_stability_accepts_bounded_drift() -> None:
    gates = ConfirmationStability(0.03, 0.05, 0.03, 0.05, 0.03, 0.02, 0.05)
    confirmation = {
        "attachment_precision": 0.90,
        "attachment_recall": 0.72,
        "attachment_f1": 0.81,
        "new_entity_detection_recall": 0.85,
        "new_entity_false_attachment_rate": 0.11,
        "overall_false_attachment_rate": 0.06,
        "manual_review_rate": 0.08,
    }
    assert passes_stability(confirmation, _development(), gates)


def test_confirmation_stability_rejects_recall_collapse() -> None:
    gates = ConfirmationStability(0.03, 0.05, 0.03, 0.05, 0.03, 0.02, 0.05)
    confirmation = _development()
    confirmation["attachment_recall"] = 0.60
    assert not passes_stability(confirmation, _development(), gates)
