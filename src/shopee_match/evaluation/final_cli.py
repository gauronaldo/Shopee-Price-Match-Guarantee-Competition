"""Command-line entry point for frozen final system evaluation."""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Sequence
from pathlib import Path

from shopee_match.evaluation.final_system_evaluator import (
    preflight_final_system_evaluation,
    run_final_system_evaluation,
)
from shopee_match.evaluation.hybrid_system_evaluator import (
    preflight_hybrid_system_evaluation,
    run_hybrid_system_evaluation,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Preflight or run final system evaluation")
    subparsers = parser.add_subparsers(dest="command", required=True)
    preflight = subparsers.add_parser("preflight", help="verify frozen inputs without test access")
    preflight.add_argument("--config", type=Path, required=True)
    evaluate = subparsers.add_parser("evaluate", help="evaluate held-out test exactly once")
    evaluate.add_argument("--config", type=Path, required=True)
    hybrid_preflight = subparsers.add_parser(
        "preflight-hybrid", help="verify the frozen hybrid system without test access"
    )
    hybrid_preflight.add_argument("--config", type=Path, required=True)
    hybrid_evaluate = subparsers.add_parser(
        "evaluate-hybrid", help="evaluate the frozen hybrid system version exactly once"
    )
    hybrid_evaluate.add_argument("--config", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    arguments = build_parser().parse_args(argv)
    if arguments.command == "preflight":
        result = preflight_final_system_evaluation(arguments.config)
        print(json.dumps(result, sort_keys=True))
        return int(result["status"] != "ready")
    if arguments.command == "evaluate":
        print(json.dumps(run_final_system_evaluation(arguments.config), sort_keys=True))
        return 0
    if arguments.command == "preflight-hybrid":
        result = preflight_hybrid_system_evaluation(arguments.config)
        print(json.dumps(result, sort_keys=True))
        return int(result["status"] != "ready")
    if arguments.command == "evaluate-hybrid":
        print(json.dumps(run_hybrid_system_evaluation(arguments.config), sort_keys=True))
        return 0
    raise AssertionError("unreachable command")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
