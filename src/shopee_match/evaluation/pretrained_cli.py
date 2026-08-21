"""Command-line entry point for Phase 9 pretrained representation benchmarks."""

from __future__ import annotations

import argparse
import importlib
import json
import logging
from collections.abc import Sequence
from pathlib import Path

import torch

from shopee_match.errors import DataValidationError
from shopee_match.evaluation.pretrained_benchmark import run_pretrained_benchmark
from shopee_match.hashing import sha256_file

WEIGHTS_FILENAME = "efficientnet_b1-c27df63c.pth"
WEIGHTS_SHA256 = "c27df63ce6eb17ef8bcea58922fd3a254cba910c720f41ee89d64d99fb7a4ddf"


def prepare_weights() -> dict[str, object]:
    """Download the official weight through TorchVision and verify its full digest."""
    try:
        models = importlib.import_module("torchvision.models")
    except ImportError as exc:
        raise DataValidationError(
            'Phase 9 requires the optional dependency: pip install -e ".[pretrained]"'
        ) from exc
    weights = models.EfficientNet_B1_Weights.IMAGENET1K_V2
    state = weights.get_state_dict(progress=True, check_hash=True)
    del state
    path = Path(torch.hub.get_dir()) / "checkpoints" / WEIGHTS_FILENAME
    actual = sha256_file(path)
    if actual != WEIGHTS_SHA256:
        raise DataValidationError(
            f"Downloaded pretrained weight digest mismatch: expected {WEIGHTS_SHA256}, got {actual}"
        )
    return {"path": str(path), "sha256": actual, "status": "verified"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run validation-only pretrained benchmarks")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser(
        "prepare-weights", help="download and verify the official EfficientNet-B1 weight"
    )
    benchmark = subparsers.add_parser("benchmark", help="evaluate frozen pretrained features")
    benchmark.add_argument("--config", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    arguments = build_parser().parse_args(argv)
    if arguments.command == "prepare-weights":
        print(json.dumps(prepare_weights(), sort_keys=True))
        return 0
    if arguments.command == "benchmark":
        print(json.dumps(run_pretrained_benchmark(arguments.config), sort_keys=True))
        return 0
    raise AssertionError("unreachable command")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
