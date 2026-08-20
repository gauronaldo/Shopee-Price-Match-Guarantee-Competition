"""Command-line interface for scratch image-embedding experiments."""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Sequence
from pathlib import Path

from shopee_match.errors import ShopeeMatchError
from shopee_match.logging import configure_logging
from shopee_match.training.image_trainer import run_scratch_image_experiment

LOGGER = logging.getLogger(__name__)


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be zero or greater")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="shopee-image")
    parser.add_argument("command", choices=("train",))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--progress-updates-per-epoch",
        type=_nonnegative_int,
        default=5,
        help="training progress messages per epoch; use 0 to disable (default: 5)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    configure_logging()
    args = build_parser().parse_args(argv)
    try:
        result = run_scratch_image_experiment(
            args.config,
            progress_updates_per_epoch=args.progress_updates_per_epoch,
        )
    except (ShopeeMatchError, OSError, ValueError, RuntimeError) as exc:
        LOGGER.error("%s", exc)
        return 2
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
