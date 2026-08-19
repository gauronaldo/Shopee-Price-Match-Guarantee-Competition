"""Seed control for deterministic Phase 0 utilities and future experiments."""

from __future__ import annotations

import os
import random
from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True, slots=True)
class SeedState:
    seed: int
    python_hash_seed: str
    numpy_seeded: bool
    torch_seeded: bool
    deterministic_algorithms: bool


def seed_everything(seed: int, *, deterministic: bool = True) -> SeedState:
    """Seed Python, NumPy, and PyTorch for repeatable local experiments."""
    if seed < 0:
        raise ValueError("seed must be non-negative")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(deterministic, warn_only=True)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = deterministic
    os.environ["PYTHONHASHSEED"] = str(seed)
    return SeedState(
        seed=seed,
        python_hash_seed=str(seed),
        numpy_seeded=True,
        torch_seeded=True,
        deterministic_algorithms=deterministic,
    )
