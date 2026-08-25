"""CLI for the scoped pair-recall development protocol."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from shopee_match.data.pair_recall_protocol import (
    load_pair_recall_protocol_config,
    write_pair_recall_protocol,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prepare the pair-recall development protocol")
    parser.add_argument("--config", type=Path, required=True)
    arguments = parser.parse_args(argv)
    config = load_pair_recall_protocol_config(arguments.config)
    print(json.dumps(write_pair_recall_protocol(config), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
