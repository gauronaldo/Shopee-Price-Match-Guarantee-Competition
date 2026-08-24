from __future__ import annotations

from shopee_match.clustering.recovery_config import _expand_fragment_attachment_grid


def test_fragment_policy_grid_expands_deterministically_with_disabled_control() -> None:
    selection = {
        "fragment_probability_thresholds": [0.1, 0.2],
        "fragment_reciprocal_rank_values": [5, 10],
        "fragment_attachment_templates": [
            {
                "maximum_source_size": 2,
                "minimum_target_size": 3,
                "minimum_support": 2,
                "minimum_source_coverage": 1.0,
                "minimum_target_support": 2,
                "target_margin": 0.02,
                "reject_variant_conflicts": True,
            }
        ],
    }
    policies = _expand_fragment_attachment_grid(selection, candidate_k=10)

    assert len(policies) == 5
    assert policies[0].enabled is False
    assert [policy.probability_threshold for policy in policies[1:]] == [0.1, 0.1, 0.2, 0.2]
    assert [policy.reciprocal_rank for policy in policies[1:]] == [5, 10, 5, 10]
