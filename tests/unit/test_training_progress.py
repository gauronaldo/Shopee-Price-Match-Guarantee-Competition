import pytest

from shopee_match.training.image_trainer import _format_duration, _progress_milestones


def test_progress_milestones_are_bounded_and_include_final_batch() -> None:
    assert _progress_milestones(850, 5) == frozenset({170, 340, 510, 680, 850})
    assert _progress_milestones(2, 5) == frozenset({1, 2})
    assert _progress_milestones(10, 0) == frozenset()


def test_progress_milestones_reject_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="total_batches"):
        _progress_milestones(0, 5)
    with pytest.raises(ValueError, match="updates"):
        _progress_milestones(10, -1)


@pytest.mark.parametrize(
    ("seconds", "expected"),
    [(0.2, "0s"), (65.0, "1m 05s"), (3661.0, "1h 01m")],
)
def test_format_duration_is_compact(seconds: float, expected: str) -> None:
    assert _format_duration(seconds) == expected
