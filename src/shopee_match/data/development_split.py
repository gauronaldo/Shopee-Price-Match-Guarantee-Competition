"""Generate a fresh development protocol while preserving the historical test split."""

from __future__ import annotations

import hashlib
import json
import os
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from shopee_match.data.development_config import DevelopmentProtocolConfig
from shopee_match.errors import DataValidationError, OutputConflictError
from shopee_match.hashing import canonical_text_sha256, sha256_file

ActiveSplit = Literal["train", "development", "confirmation"]
ProtocolSplit = Literal["train", "development", "confirmation", "historical_test"]
ACTIVE_SPLITS: tuple[ActiveSplit, ...] = ("train", "development", "confirmation")


@dataclass(frozen=True, slots=True)
class SourceManifestRow:
    posting_id: str
    image: str
    label_group: str
    source_split: str
    super_component_id: str


@dataclass(frozen=True, slots=True)
class ProtocolComponent:
    component_id: str
    posting_ids: tuple[str, ...]
    label_groups: tuple[str, ...]
    row_count: int
    size_band: str


def _size_band(maximum_group_size: int) -> str:
    if maximum_group_size <= 2:
        return "2"
    if maximum_group_size <= 5:
        return "3_to_5"
    if maximum_group_size <= 9:
        return "6_to_9"
    return "10_plus"


def load_source_manifest(path: Path) -> tuple[SourceManifestRow, ...]:
    """Load the immutable v1 manifest without joining raw metadata."""
    rows: list[SourceManifestRow] = []
    posting_ids: set[str] = set()
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                payload = json.loads(line)
                expected = {
                    "posting_id",
                    "image",
                    "label_group",
                    "split",
                    "super_component_id",
                }
                if set(payload) != expected:
                    raise DataValidationError(
                        f"Source manifest row {line_number} has an unexpected schema"
                    )
                posting_id = str(payload["posting_id"])
                if posting_id in posting_ids:
                    raise DataValidationError(f"Duplicate source posting_id: {posting_id}")
                posting_ids.add(posting_id)
                source_split = str(payload["split"])
                if source_split not in {"train", "validation", "test"}:
                    raise DataValidationError(f"Unknown source split: {source_split}")
                rows.append(
                    SourceManifestRow(
                        posting_id=posting_id,
                        image=str(payload["image"]),
                        label_group=str(payload["label_group"]),
                        source_split=source_split,
                        super_component_id=str(payload["super_component_id"]),
                    )
                )
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise DataValidationError("Cannot load the source split manifest") from exc
    if not rows:
        raise DataValidationError("Source split manifest is empty")
    return tuple(rows)


def build_active_components(rows: tuple[SourceManifestRow, ...]) -> tuple[ProtocolComponent, ...]:
    """Reuse v1 leakage super-components for all non-test rows."""
    active = [row for row in rows if row.source_split != "test"]
    group_sizes = Counter(row.label_group for row in active)
    by_component: dict[str, list[SourceManifestRow]] = defaultdict(list)
    for row in active:
        by_component[row.super_component_id].append(row)
    components = []
    for component_id, members in by_component.items():
        labels = tuple(sorted({row.label_group for row in members}))
        maximum_group_size = max(group_sizes[label] for label in labels)
        components.append(
            ProtocolComponent(
                component_id=component_id,
                posting_ids=tuple(sorted(row.posting_id for row in members)),
                label_groups=labels,
                row_count=len(members),
                size_band=_size_band(maximum_group_size),
            )
        )
    if not components:
        raise DataValidationError("No active components remain after preserving historical test")
    return tuple(sorted(components, key=lambda component: component.component_id))


def assign_active_components(
    components: tuple[ProtocolComponent, ...], config: DevelopmentProtocolConfig
) -> dict[str, ActiveSplit]:
    """Assign whole components by deterministic normalized load balancing."""
    allocation = config.allocation
    fractions = {
        "train": allocation.train_fraction,
        "development": allocation.development_fraction,
        "confirmation": allocation.confirmation_fraction,
    }
    total_rows = sum(component.row_count for component in components)
    band_totals = Counter(component.size_band for component in components)
    assigned_rows: Counter[str] = Counter()
    assigned_bands: dict[str, Counter[str]] = defaultdict(Counter)
    assignments: dict[str, ActiveSplit] = {}

    def stable_order(component: ProtocolComponent) -> tuple[int, str]:
        material = f"{allocation.seed}:{component.component_id}".encode()
        return -component.row_count, hashlib.sha256(material).hexdigest()

    for component in sorted(components, key=stable_order):
        scored: list[tuple[float, int, ActiveSplit]] = []
        for index, split in enumerate(ACTIVE_SPLITS):
            row_target = total_rows * fractions[split]
            band_target = band_totals[component.size_band] * fractions[split]
            row_load = assigned_rows[split] / row_target
            band_load = assigned_bands[component.size_band][split] / max(band_target, 1e-12)
            scored.append((0.7 * row_load + 0.3 * band_load, index, split))
        selected = min(scored)[2]
        assignments[component.component_id] = selected
        assigned_rows[selected] += component.row_count
        assigned_bands[component.size_band][selected] += 1
    if set(assignments.values()) != set(ACTIVE_SPLITS):
        raise DataValidationError("Development allocation produced an empty active split")
    return assignments


def create_development_protocol(
    rows: tuple[SourceManifestRow, ...], config: DevelopmentProtocolConfig
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    """Create and audit the four-way protocol without tuning on historical test."""
    components = build_active_components(rows)
    component_assignments = assign_active_components(components, config)
    output: list[dict[str, str]] = []
    for row in rows:
        split: ProtocolSplit = (
            "historical_test"
            if row.source_split == config.source.preserved_split
            else component_assignments[row.super_component_id]
        )
        output.append(
            {
                "image": row.image,
                "label_group": row.label_group,
                "posting_id": row.posting_id,
                "source_split": row.source_split,
                "split": split,
                "super_component_id": row.super_component_id,
            }
        )
    output.sort(key=lambda row: row["posting_id"])
    labels: dict[str, set[str]] = defaultdict(set)
    super_components: dict[str, set[str]] = defaultdict(set)
    for manifest_row in output:
        labels[manifest_row["label_group"]].add(manifest_row["split"])
        super_components[manifest_row["super_component_id"]].add(manifest_row["split"])
    source_test_ids = {row.posting_id for row in rows if row.source_split == "test"}
    historical_test_ids = {
        row["posting_id"] for row in output if row["split"] == "historical_test"
    }
    integrity = {
        "label_groups_cross_split": sum(len(splits) > 1 for splits in labels.values()),
        "super_components_cross_split": sum(
            len(splits) > 1 for splits in super_components.values()
        ),
        "historical_test_id_symmetric_difference": len(
            source_test_ids.symmetric_difference(historical_test_ids)
        ),
    }
    if any(integrity.values()):
        raise DataValidationError(f"Development protocol integrity failure: {integrity}")
    summary = {
        "protocol_version": "entity_resolution.development_protocol.v2",
        "seed": config.allocation.seed,
        "source_manifest_sha256": config.source.manifest_sha256,
        "config_sha256": canonical_text_sha256(config.config_path),
        "listings": dict(sorted(Counter(row["split"] for row in output).items())),
        "label_groups": dict(
            sorted(
                Counter(
                    next(iter(splits))
                    for splits in labels.values()
                ).items()
            )
        ),
        "super_components": dict(
            sorted(
                Counter(
                    next(iter(splits))
                    for splits in super_components.values()
                ).items()
            )
        ),
        "active_super_components": len(components),
        "historical_test_preserved": True,
        "historical_test_used_for_selection": False,
        "integrity": integrity,
    }
    return output, summary


def _write_atomic(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def modeling_manifest(manifest: list[dict[str, str]]) -> list[dict[str, str]]:
    """Map v2 roles to the legacy trainer contract without exposing confirmation separately."""
    role_mapping = {
        "train": "train",
        "development": "validation",
        "confirmation": "test",
        "historical_test": "test",
    }
    return [
        {
            "posting_id": row["posting_id"],
            "split": role_mapping[row["split"]],
        }
        for row in manifest
    ]


def write_development_protocol(config: DevelopmentProtocolConfig) -> dict[str, object]:
    """Write immutable protocol evidence and return its identifying hashes."""
    outputs = (
        config.artifacts.manifest_path,
        config.artifacts.modeling_manifest_path,
        config.artifacts.summary_path,
    )
    existing = [str(path) for path in outputs if path.exists()]
    if existing:
        raise OutputConflictError(
            "Refusing to overwrite development protocol evidence: " + ", ".join(existing)
        )
    rows = load_source_manifest(config.source.manifest_path)
    manifest, summary = create_development_protocol(rows, config)
    content = "".join(json.dumps(row, sort_keys=True) + "\n" for row in manifest)
    model_content = "".join(
        json.dumps(row, sort_keys=True) + "\n" for row in modeling_manifest(manifest)
    )
    _write_atomic(config.artifacts.manifest_path, content)
    _write_atomic(config.artifacts.modeling_manifest_path, model_content)
    summary_content = json.dumps(summary, indent=2, sort_keys=True) + "\n"
    _write_atomic(config.artifacts.summary_path, summary_content)
    return {
        "status": "development_protocol_v2_complete",
        "manifest": str(config.artifacts.manifest_path),
        "manifest_sha256": sha256_file(config.artifacts.manifest_path),
        "modeling_manifest": str(config.artifacts.modeling_manifest_path),
        "modeling_manifest_sha256": sha256_file(config.artifacts.modeling_manifest_path),
        "summary": str(config.artifacts.summary_path),
        "splits": summary["listings"],
        "historical_test_preserved": True,
        "historical_test_used_for_selection": False,
    }
