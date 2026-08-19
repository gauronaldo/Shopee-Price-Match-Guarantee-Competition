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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="shopee-image")
    parser.add_argument("command", choices=("train",))
    parser.add_argument("--config", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    configure_logging()
    args = build_parser().parse_args(argv)
    try:
        result = run_scratch_image_experiment(args.config)
    except (ShopeeMatchError, OSError, ValueError, RuntimeError) as exc:
        LOGGER.error("%s", exc)
        return 2
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
