"""Deterministic query/catalog role construction for catalog-attachment evaluation."""

from __future__ import annotations

import hashlib
import json
import os
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from shopee_match.errors import ConfigurationError, DataValidationError, OutputConflictError
from shopee_match.evaluation.protocol import EvaluationSplit, load_named_split
from shopee_match.hashing import canonical_text_sha256, sha256_file
from shopee_match.training.multimodal_trainer import _git_state
from shopee_match.training.text_config import (
    _mapping,
    _number,
    _only_keys,
    _positive_int,
    _read_yaml,
    _relative_path,
    _typed,
)


@dataclass(frozen=True, slots=True)
class CatalogRole:
    posting_id: str
    partition: str
    role: str


@dataclass(frozen=True, slots=True)
class CatalogProtocolConfig:
    seed: int
    metadata_csv: Path
    source_manifest: Path
    source_split: str
    partition: str
    known_entity_fraction: float
    references_per_known_entity: int
    manifest: Path
    summary: Path
    config_path: Path


def _verified(raw: dict[str, Any], name: str, *, portable_text: bool = False) -> Path:
    path = _relative_path(raw[name], f"source.{name}")
    expected = _typed(raw[f"{name}_sha256"], str, f"source.{name}_sha256").lower()
    actual = canonical_text_sha256(path) if portable_text else sha256_file(path)
    if actual != expected:
        raise ConfigurationError(
            f"Catalog protocol source mismatch for {path}: expected {expected}, got {actual}"
        )
    return path


def load_catalog_protocol_config(path: Path) -> CatalogProtocolConfig:
    root = _read_yaml(path, "catalog-attachment protocol config")
    _only_keys(root, {"config_version", "seed", "source", "roles", "artifacts"}, "config")
    if root["config_version"] != "catalog_attachment.protocol.v4":
        raise ConfigurationError("Unsupported catalog-attachment protocol version")

    source = _mapping(root["source"], "source")
    _only_keys(
        source,
        {
            "metadata_csv",
            "metadata_csv_sha256",
            "source_manifest",
            "source_manifest_sha256",
            "source_split",
            "partition",
        },
        "source",
    )
    metadata = _verified(source, "metadata_csv")
    source_manifest = _verified(source, "source_manifest")
    source_split = _typed(source["source_split"], str, "source.source_split")
    partition = _typed(source["partition"], str, "source.partition")
    if source_split != "validation" or partition != "development":
        raise ConfigurationError(
            "Protocol v4 development construction must use only the validation source split"
        )

    roles = _mapping(root["roles"], "roles")
    _only_keys(
        roles,
        {"known_entity_fraction", "references_per_known_entity"},
        "roles",
    )
    known_fraction = _number(roles["known_entity_fraction"], "roles.known_entity_fraction")
    if not 0.0 < known_fraction < 1.0:
        raise ConfigurationError("known_entity_fraction must be inside (0, 1)")
    references = _positive_int(
        roles["references_per_known_entity"], "roles.references_per_known_entity"
    )
    if references != 1:
        raise ConfigurationError("Protocol v4 currently requires one catalog reference per entity")

    artifacts = _mapping(root["artifacts"], "artifacts")
    _only_keys(artifacts, {"manifest", "summary"}, "artifacts")
    return CatalogProtocolConfig(
        seed=_positive_int(root["seed"], "seed"),
        metadata_csv=metadata,
        source_manifest=source_manifest,
        source_split=source_split,
        partition=partition,
        known_entity_fraction=known_fraction,
        references_per_known_entity=references,
        manifest=_relative_path(artifacts["manifest"], "artifacts.manifest"),
        summary=_relative_path(artifacts["summary"], "artifacts.summary"),
        config_path=path,
    )


def _stable_key(seed: int, namespace: str, value: str) -> str:
    return hashlib.sha256(f"{seed}:{namespace}:{value}".encode()).hexdigest()


def assign_catalog_roles(
    split: EvaluationSplit,
    *,
    seed: int,
    partition: str,
    known_entity_fraction: float,
) -> tuple[CatalogRole, ...]:
    """Assign disjoint catalog, known-query, and new-query roles using labels only here."""
    groups: dict[str, list[str]] = defaultdict(list)
    for item in split.items:
        groups[split.label_by_id[item.posting_id]].append(item.posting_id)
    eligible = [label for label, identifiers in groups.items() if len(identifiers) >= 2]
    eligible.sort(key=lambda label: (_stable_key(seed, "known-group", label), label))
    known_count = min(len(eligible) - 1, max(1, round(len(eligible) * known_entity_fraction)))
    known_labels = set(eligible[:known_count])

    roles: list[CatalogRole] = []
    for label in sorted(groups):
        identifiers = sorted(groups[label])
        if label not in known_labels:
            roles.extend(
                CatalogRole(identifier, partition, "new_entity_query") for identifier in identifiers
            )
            continue
        reference = min(
            identifiers,
            key=lambda identifier: (_stable_key(seed, "reference", identifier), identifier),
        )
        roles.append(CatalogRole(reference, partition, "catalog_reference"))
        roles.extend(
            CatalogRole(identifier, partition, "known_entity_query")
            for identifier in identifiers
            if identifier != reference
        )
    result = tuple(sorted(roles, key=lambda row: row.posting_id))
    if len(result) != len(split.items) or len({row.posting_id for row in result}) != len(result):
        raise DataValidationError("Catalog role assignment is incomplete or duplicated")
    return result


def validate_catalog_roles(split: EvaluationSplit, roles: tuple[CatalogRole, ...]) -> None:
    role_by_id = {row.posting_id: row.role for row in roles}
    if set(role_by_id) != set(split.label_by_id):
        raise DataValidationError("Catalog roles do not cover exactly the source split")
    reference_labels = {
        split.label_by_id[posting_id]
        for posting_id, role in role_by_id.items()
        if role == "catalog_reference"
    }
    for posting_id, role in role_by_id.items():
        label = split.label_by_id[posting_id]
        if role == "known_entity_query" and label not in reference_labels:
            raise DataValidationError("Known-entity query has no catalog reference")
        if role == "new_entity_query" and label in reference_labels:
            raise DataValidationError("New-entity query leaks its label into the catalog")
    reference_counts: dict[str, int] = defaultdict(int)
    for posting_id, role in role_by_id.items():
        if role == "catalog_reference":
            reference_counts[split.label_by_id[posting_id]] += 1
    if any(count != 1 for count in reference_counts.values()):
        raise DataValidationError("Known entities must have exactly one catalog reference")


def _write_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def build_catalog_protocol(config_path: Path) -> dict[str, object]:
    config = load_catalog_protocol_config(config_path)
    existing = [str(path) for path in (config.manifest, config.summary) if path.exists()]
    if existing:
        raise OutputConflictError(
            "Refusing to overwrite catalog protocol outputs: " + ", ".join(existing)
        )
    commit, dirty = _git_state()
    if dirty:
        raise DataValidationError("Catalog protocol construction requires a clean Git worktree")
    split = load_named_split(config.metadata_csv, config.source_manifest, config.source_split)
    roles = assign_catalog_roles(
        split,
        seed=config.seed,
        partition=config.partition,
        known_entity_fraction=config.known_entity_fraction,
    )
    validate_catalog_roles(split, roles)
    counts: dict[str, int] = defaultdict(int)
    for row in roles:
        counts[row.role] += 1
    manifest_text = "".join(json.dumps(asdict(row), sort_keys=True) + "\n" for row in roles)
    _write_atomic(config.manifest, manifest_text)
    summary = {
        "protocol_version": "catalog_attachment.protocol.v4",
        "status": "development_roles_complete",
        "provenance": {
            "git_commit": commit,
            "git_dirty": False,
            "config_sha256": canonical_text_sha256(config.config_path),
            "source_manifest_sha256": sha256_file(config.source_manifest),
        },
        "partition": config.partition,
        "source_split": config.source_split,
        "listings": len(roles),
        "role_counts": dict(sorted(counts.items())),
        "catalog_entities": counts["catalog_reference"],
        "confirmation_accessed": False,
        "labels_persisted_in_role_manifest": False,
    }
    _write_atomic(config.summary, json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return {
        "status": summary["status"],
        "manifest": str(config.manifest),
        "summary": str(config.summary),
        "role_counts": summary["role_counts"],
        "confirmation_accessed": False,
    }


def load_catalog_roles(path: Path, *, partition: str) -> tuple[CatalogRole, ...]:
    roles: list[CatalogRole] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                raw = json.loads(line)
                row = CatalogRole(
                    posting_id=str(raw["posting_id"]),
                    partition=str(raw["partition"]),
                    role=str(raw["role"]),
                )
            except (json.JSONDecodeError, KeyError, TypeError) as error:
                raise DataValidationError(
                    f"Invalid catalog role record at line {line_number}"
                ) from error
            if row.partition != partition:
                raise DataValidationError("Catalog role manifest contains an unexpected partition")
            if row.role not in {
                "catalog_reference",
                "known_entity_query",
                "new_entity_query",
            }:
                raise DataValidationError(f"Unsupported catalog role: {row.role}")
            roles.append(row)
    if not roles or len({row.posting_id for row in roles}) != len(roles):
        raise DataValidationError("Catalog role manifest is empty or contains duplicate IDs")
    return tuple(roles)
