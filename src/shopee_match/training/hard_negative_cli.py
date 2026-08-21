"""Command-line interface for Phase 6 hard-negative mining and fine-tuning."""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Sequence
from pathlib import Path

from shopee_match.errors import ShopeeMatchError
from shopee_match.logging import configure_logging
from shopee_match.training.hard_negative_analyzer import summarize_hard_negative_runs
from shopee_match.training.hard_negative_miner import mine_hard_negatives
from shopee_match.training.hard_negative_trainer import run_hard_negative_experiment

LOGGER = logging.getLogger(__name__)


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be zero or greater")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="shopee-hard-negatives")
    parser.add_argument("command", choices=("mine", "train", "all", "summarize"))
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
        if args.command == "mine":
            result: dict[str, object] = mine_hard_negatives(args.config)
        elif args.command == "train":
            result = run_hard_negative_experiment(
                args.config,
                progress_updates_per_epoch=args.progress_updates_per_epoch,
            )
        elif args.command == "all":
            mining = mine_hard_negatives(args.config)
            training = run_hard_negative_experiment(
                args.config,
                progress_updates_per_epoch=args.progress_updates_per_epoch,
            )
            result = {"status": training["status"], "mining": mining, "training": training}
        else:
            result = summarize_hard_negative_runs(args.config)
    except (ShopeeMatchError, OSError, ValueError, RuntimeError) as exc:
        LOGGER.error("%s", exc)
        return 2
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
