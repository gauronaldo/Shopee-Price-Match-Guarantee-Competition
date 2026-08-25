"""CLI for the catalog-attachment evaluation protocol."""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Sequence
from pathlib import Path

from shopee_match.errors import ShopeeMatchError
from shopee_match.evaluation.catalog_attachment_evaluator import (
    run_catalog_attachment_evaluation,
)
from shopee_match.evaluation.catalog_attachment_protocol import build_catalog_protocol
from shopee_match.logging import configure_logging

LOGGER = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="shopee-catalog-attachment")
    parser.add_argument("command", choices=("build-protocol", "evaluate"))
    parser.add_argument("--config", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    configure_logging()
    args = build_parser().parse_args(argv)
    try:
        result = (
            build_catalog_protocol(args.config)
            if args.command == "build-protocol"
            else run_catalog_attachment_evaluation(args.config)
        )
    except (ShopeeMatchError, OSError, ValueError, RuntimeError) as error:
        LOGGER.error("%s", error)
        return 2
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
