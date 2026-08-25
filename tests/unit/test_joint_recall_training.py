from __future__ import annotations

from shopee_match.training.joint_recall_trainer import JointRecallConfig


def test_joint_recall_config_records_tiered_learning_rates() -> None:
    fields = JointRecallConfig.__dataclass_fields__
    assert "learning_rates" in fields
    assert "thresholds" in fields
    assert "gates" in fields
