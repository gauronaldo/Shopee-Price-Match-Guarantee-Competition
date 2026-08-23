"""Command-line entry point for Phase 8 entity resolution."""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Sequence
from pathlib import Path

from shopee_match.clustering.benchmark import run_entity_resolution_benchmark
from shopee_match.clustering.hybrid_entity import run_hybrid_entity_evaluation
from shopee_match.clustering.pair_evidence_graph import run_pair_evidence_graph_selection
from shopee_match.clustering.recall_recovery import run_entity_recall_recovery
from shopee_match.training.pair_evidence_trainer import run_pair_evidence_experiment


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run validation-only entity resolution")
    subparsers = parser.add_subparsers(dest="command", required=True)
    benchmark = subparsers.add_parser("benchmark", help="score pairs and select graph policy")
    benchmark.add_argument("--config", type=Path, required=True)
    recover = subparsers.add_parser(
        "recover-recall",
        help="sweep singleton recovery policies on frozen validation pair scores",
    )
    recover.add_argument("--config", type=Path, required=True)
    evidence = subparsers.add_parser(
        "train-pair-evidence",
        help="train a residual classical-evidence head with frozen multimodal scores",
    )
    evidence.add_argument("--config", type=Path, required=True)
    evidence_graph = subparsers.add_parser(
        "select-pair-evidence-graph",
        help="select graph thresholds for frozen pair-evidence scores",
    )
    evidence_graph.add_argument("--config", type=Path, required=True)
    hybrid = subparsers.add_parser(
        "evaluate-hybrid-candidates",
        help="score frozen hybrid candidates and evaluate entity clustering",
    )
    hybrid.add_argument("--config", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    arguments = build_parser().parse_args(argv)
    if arguments.command == "benchmark":
        result = run_entity_resolution_benchmark(arguments.config)
        print(json.dumps(result, sort_keys=True))
        return 0
    if arguments.command == "recover-recall":
        result = run_entity_recall_recovery(arguments.config)
        print(json.dumps(result, sort_keys=True))
        return 0
    if arguments.command == "train-pair-evidence":
        result = run_pair_evidence_experiment(arguments.config)
        print(json.dumps(result, sort_keys=True))
        return 0
    if arguments.command == "select-pair-evidence-graph":
        result = run_pair_evidence_graph_selection(arguments.config)
        print(json.dumps(result, sort_keys=True))
        return 0
    if arguments.command == "evaluate-hybrid-candidates":
        result = run_hybrid_entity_evaluation(arguments.config)
        print(json.dumps(result, sort_keys=True))
        return 0
    raise AssertionError("unreachable command")


if __name__ == "__main__":  # pragma: no cover - exercised through the console entry point
    raise SystemExit(main())
