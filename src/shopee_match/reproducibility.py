"""Seed control for deterministic Phase 0 utilities and future experiments."""

from __future__ import annotations

import os
import random
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class SeedState:
    seed: int
    python_hash_seed: str
    numpy_seeded: bool


def seed_everything(seed: int) -> SeedState:
    """Seed current Phase 0 RNGs; later PyTorch phases must extend and test this function."""
    if seed < 0:
        raise ValueError("seed must be non-negative")
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    return SeedState(seed=seed, python_hash_seed=str(seed), numpy_seeded=True)
