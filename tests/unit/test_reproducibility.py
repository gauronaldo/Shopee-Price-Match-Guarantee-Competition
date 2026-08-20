import random

import numpy as np
import pytest

from shopee_match.reproducibility import seed_everything


def test_seed_everything_repeats_python_and_numpy_sequences() -> None:
    first_state = seed_everything(42)
    first = (random.random(), float(np.random.random()))
    second_state = seed_everything(42)
    second = (random.random(), float(np.random.random()))

    assert first_state == second_state
    assert first == second


def test_deterministic_seed_configures_cublas_before_cuda_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)

    state = seed_everything(7, deterministic=True)

    assert state.cublas_workspace_config == ":4096:8"
