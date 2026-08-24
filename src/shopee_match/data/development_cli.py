"""CLI for the version-two entity-resolution development protocol."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from shopee_match.data.development_config import load_development_protocol_config
from shopee_match.data.development_split import write_development_protocol


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare the v2 development protocol")
    parser.add_argument("--config", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    config = load_development_protocol_config(arguments.config)
    print(json.dumps(write_development_protocol(config), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
