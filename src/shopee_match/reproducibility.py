"""Seed and backend controls for reproducible CPU and CUDA experiments."""

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
    cublas_workspace_config: str | None


def seed_everything(seed: int, *, deterministic: bool = True) -> SeedState:
    """Seed Python, NumPy, and PyTorch for repeatable local experiments."""
    if seed < 0:
        raise ValueError("seed must be non-negative")
    if deterministic:
        # CUDA >= 10.2 requires this before the first CuBLAS operation. Without it,
        # PyTorch can only warn that linear layers and contrastive matrix products
        # are nondeterministic even when deterministic algorithms are requested.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():  # type: ignore[no-untyped-call]
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
        cublas_workspace_config=os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    )
