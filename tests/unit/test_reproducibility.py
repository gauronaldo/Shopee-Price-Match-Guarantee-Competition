import random

import numpy as np

from shopee_match.reproducibility import seed_everything


def test_seed_everything_repeats_python_and_numpy_sequences() -> None:
    first_state = seed_everything(42)
    first = (random.random(), float(np.random.random()))
    second_state = seed_everything(42)
    second = (random.random(), float(np.random.random()))

    assert first_state == second_state
    assert first == second
