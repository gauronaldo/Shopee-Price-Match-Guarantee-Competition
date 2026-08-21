"""Command-line interface for Phase 5 scratch multimodal experiments."""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Sequence
from pathlib import Path

from shopee_match.errors import ShopeeMatchError
from shopee_match.logging import configure_logging
from shopee_match.training.multimodal_data import prepare_frozen_multimodal_cache
from shopee_match.training.multimodal_trainer import (
    refresh_multimodal_training_report,
    run_multimodal_experiment,
)

LOGGER = logging.getLogger(__name__)


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be zero or greater")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="shopee-multimodal")
    parser.add_argument("command", choices=("prepare", "train", "refresh-report"))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--progress-updates-per-epoch",
        type=_nonnegative_int,
        default=4,
        help="bounded training updates per epoch; use 0 to disable (default: 4)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    configure_logging()
    args = build_parser().parse_args(argv)
    try:
        if args.command == "prepare":
            result = prepare_frozen_multimodal_cache(args.config)
        elif args.command == "train":
            result = run_multimodal_experiment(
                args.config,
                progress_updates_per_epoch=args.progress_updates_per_epoch,
            )
        else:
            result = refresh_multimodal_training_report(args.config)
    except (ShopeeMatchError, OSError, ValueError, RuntimeError) as exc:
        LOGGER.error("%s", exc)
        return 2
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
