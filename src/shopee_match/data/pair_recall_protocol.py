"""Scoped development protocol for post-confirmation pair-recall experiments."""

from __future__ import annotations

import hashlib
import json
import os
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from shopee_match.errors import ConfigurationError, DataValidationError, OutputConflictError
from shopee_match.hashing import canonical_text_sha256, sha256_file
from shopee_match.training.text_config import (
    _mapping,
    _nonnegative_int,
    _only_keys,
    _read_yaml,
    _relative_path,
    _typed,
)


@dataclass(frozen=True, slots=True)
class PairRecallProtocolConfig:
    source_manifest: Path
    source_sha256: str
    train_role: str
    repartition_role: str
    seed: int
    development_fraction: float
    manifest: Path
    summary: Path
    config_path: Path


def load_pair_recall_protocol_config(path: Path) -> PairRecallProtocolConfig:
    """Load a protocol that never exposes v2 confirmation or historical-test rows."""
    root = _read_yaml(path, "pair-recall development protocol")
    _only_keys(root, {"config_version", "source", "allocation", "artifacts"}, "config")
    if root["config_version"] != "pair_recall.development_protocol.v1":
        raise ConfigurationError("Unsupported pair-recall protocol version")
    source = _mapping(root["source"], "source")
    _only_keys(
        source,
        {"manifest", "manifest_sha256", "train_role", "repartition_role"},
        "source",
    )
    source_manifest = _relative_path(source["manifest"], "source.manifest")
    source_sha = _typed(source["manifest_sha256"], str, "source.manifest_sha256").lower()
    if len(source_sha) != 64 or sha256_file(source_manifest) != source_sha:
        raise ConfigurationError("Pair-recall source manifest SHA-256 mismatch")
    train_role = _typed(source["train_role"], str, "source.train_role")
    repartition_role = _typed(source["repartition_role"], str, "source.repartition_role")
    if (train_role, repartition_role) != ("train", "development"):
        raise ConfigurationError("Pair-recall protocol must use v2 train and development only")
    allocation = _mapping(root["allocation"], "allocation")
    _only_keys(allocation, {"seed", "development_fraction"}, "allocation")
    fraction = allocation["development_fraction"]
    if isinstance(fraction, bool) or not isinstance(fraction, int | float):
        raise ConfigurationError("allocation.development_fraction must be numeric")
    fraction = float(fraction)
    if not 0.0 < fraction < 1.0:
        raise ConfigurationError("allocation.development_fraction must be inside (0, 1)")
    artifacts = _mapping(root["artifacts"], "artifacts")
    _only_keys(artifacts, {"manifest", "summary"}, "artifacts")
    manifest = _relative_path(artifacts["manifest"], "artifacts.manifest")
    summary = _relative_path(artifacts["summary"], "artifacts.summary")
    if manifest == source_manifest or manifest == summary:
        raise ConfigurationError("Pair-recall protocol outputs must be distinct")
    return PairRecallProtocolConfig(
        source_manifest,
        source_sha,
        train_role,
        repartition_role,
        _nonnegative_int(allocation["seed"], "allocation.seed"),
        fraction,
        manifest,
        summary,
        path,
    )


def _read_rows(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    posting_ids: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            payload = json.loads(line)
            required = {
                "posting_id",
                "label_group",
                "split",
                "super_component_id",
            }
            if not required.issubset(payload):
                raise DataValidationError(
                    f"Pair-recall source row {line_number} is missing required fields"
                )
            posting_id = str(payload["posting_id"])
            if posting_id in posting_ids:
                raise DataValidationError(f"Duplicate posting_id in source: {posting_id}")
            posting_ids.add(posting_id)
            rows.append({key: str(value) for key, value in payload.items()})
    if not rows:
        raise DataValidationError("Pair-recall source manifest is empty")
    return rows


def create_pair_recall_protocol(
    rows: list[dict[str, str]], config: PairRecallProtocolConfig
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    """Keep train intact and split whole v2-development super-components in half."""
    training = [row for row in rows if row["split"] == config.train_role]
    development = [row for row in rows if row["split"] == config.repartition_role]
    excluded = [
        row
        for row in rows
        if row["split"] not in {config.train_role, config.repartition_role}
    ]
    by_component: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in development:
        by_component[row["super_component_id"]].append(row)
    if not training or len(by_component) < 2:
        raise DataValidationError("Pair-recall protocol requires train and development components")
    total = len(development)
    target = total * config.development_fraction
    development_rows = 0
    component_role: dict[str, str] = {}
    ordered = sorted(
        by_component,
        key=lambda component: (
            -len(by_component[component]),
            hashlib.sha256(f"{config.seed}:{component}".encode()).hexdigest(),
        ),
    )
    for component in ordered:
        size = len(by_component[component])
        before = abs(development_rows - target)
        after = abs(development_rows + size - target)
        role = "validation" if after <= before else "test"
        component_role[component] = role
        if role == "validation":
            development_rows += size
    output = [
        {"posting_id": row["posting_id"], "split": "train"} for row in training
    ] + [
        {
            "posting_id": row["posting_id"],
            "split": component_role[row["super_component_id"]],
        }
        for row in development
    ]
    output.sort(key=lambda row: row["posting_id"])
    output_ids = {row["posting_id"] for row in output}
    labels: dict[str, set[str]] = defaultdict(set)
    components: dict[str, set[str]] = defaultdict(set)
    row_by_id = {row["posting_id"]: row for row in rows}
    for row in output:
        source = row_by_id[row["posting_id"]]
        labels[source["label_group"]].add(row["split"])
        components[source["super_component_id"]].add(row["split"])
    integrity = {
        "label_groups_cross_split": sum(len(roles) > 1 for roles in labels.values()),
        "super_components_cross_split": sum(
            len(roles) > 1 for roles in components.values()
        ),
        "v2_confirmation_rows_exposed": sum(
            row["split"] == "confirmation" and row["posting_id"] in output_ids
            for row in excluded
        ),
    }
    if any(integrity.values()):
        raise DataValidationError(f"Pair-recall protocol integrity failure: {integrity}")
    summary = {
        "protocol_version": "pair_recall.development_protocol.v1",
        "seed": config.seed,
        "source_manifest_sha256": config.source_sha256,
        "config_sha256": canonical_text_sha256(config.config_path),
        "listings": dict(sorted(Counter(row["split"] for row in output).items())),
        "label_groups": dict(
            sorted(Counter(next(iter(roles)) for roles in labels.values()).items())
        ),
        "excluded_source_roles": dict(
            sorted(Counter(row["split"] for row in excluded).items())
        ),
        "integrity": integrity,
        "confirmation_v2_accessed": False,
        "historical_test_accessed": False,
        "limitation": (
            "The internal confirmation is a previously untrained subset of the v2 development "
            "pool, whose aggregate metrics were observed before this protocol was created."
        ),
    }
    return output, summary


def _write_atomic(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def write_pair_recall_protocol(config: PairRecallProtocolConfig) -> dict[str, object]:
    """Write immutable scoped manifests for joint recall development."""
    existing = [str(path) for path in (config.manifest, config.summary) if path.exists()]
    if existing:
        raise OutputConflictError(
            "Refusing to overwrite pair-recall protocol: " + ", ".join(existing)
        )
    output, summary = create_pair_recall_protocol(_read_rows(config.source_manifest), config)
    _write_atomic(
        config.manifest,
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in output),
    )
    _write_atomic(config.summary, json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return {
        "status": "pair_recall_development_protocol_complete",
        "manifest": str(config.manifest),
        "manifest_sha256": sha256_file(config.manifest),
        "summary": str(config.summary),
        "splits": summary["listings"],
        "confirmation_v2_accessed": False,
        "historical_test_accessed": False,
    }
