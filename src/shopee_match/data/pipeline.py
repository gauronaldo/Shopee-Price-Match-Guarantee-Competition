"""Phase 1 orchestration from immutable Kaggle source to audited split artifacts."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from shopee_match.data.audit import AuditBundle, audit_dataset
from shopee_match.data.config import Phase1Config, load_phase1_config
from shopee_match.data.io import audit_images, load_listings, sha256_file
from shopee_match.data.models import Listing, SplitName
from shopee_match.data.report import write_reports
from shopee_match.data.split import SplitBundle, create_split
from shopee_match.errors import DataValidationError, OutputConflictError


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _immutable_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() == content:
            return
        raise OutputConflictError(
            f"Versioned output {path} already exists with different content; "
            "select a new versioned path or remove it after review"
        )
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(content)
    temporary.replace(path)


def _manifest_bytes(listings: list[Listing], split: SplitBundle) -> bytes:
    lines: list[bytes] = []
    for item in sorted(listings, key=lambda listing: listing.posting_id):
        component_id = split.component_ids[item.label_group]
        record = {
            "posting_id": item.posting_id,
            "image": item.image,
            "label_group": item.label_group,
            "super_component_id": component_id,
            "split": split.label_assignments[item.label_group],
        }
        lines.append(json.dumps(record, sort_keys=True).encode("utf-8") + b"\n")
    return b"".join(lines)


def _near_phash_cross_split(
    listings: list[Listing], audit: AuditBundle, assignments: dict[str, SplitName]
) -> int:
    phash_splits: dict[str, set[SplitName]] = defaultdict(set)
    for item in listings:
        phash_splits[item.image_phash].add(assignments[item.label_group])
    return sum(
        len(phash_splits[left] | phash_splits[right]) > 1
        for left, right, _ in audit.near_phash_pairs
    )


def _provenance(config: Phase1Config, manifest_sha256: str, metadata_sha256: str) -> dict[str, Any]:
    return {
        "pipeline_version": "phase1.pipeline.v1",
        "config_version": config.config_version,
        "config_sha256": sha256_file(config.config_path),
        "metadata_sha256": metadata_sha256,
        "manifest_sha256": manifest_sha256,
        "competition_slug": config.dataset.competition_slug,
    }


def prepare_dataset(config_path: Path) -> dict[str, Any]:
    """Audit real data, create a leakage-safe split, and write reproducible artifacts."""
    config = load_phase1_config(config_path)
    listings, findings = load_listings(config)
    images, image_findings, image_stats = audit_images(listings, config.dataset.image_dir)
    findings.extend(image_findings)
    audit = audit_dataset(
        listings,
        images,
        image_stats,
        config.audit.near_phash_hamming_distance,
        findings,
    )
    critical = [item for item in audit.findings if item.severity == "critical"]
    if critical:
        details = ", ".join(f"{item.code}={item.count}" for item in critical)
        raise DataValidationError(f"Critical audit findings block split generation: {details}")

    split = create_split(listings, images, config.split)
    split.summary["near_phash_pairs_cross_split"] = _near_phash_cross_split(
        listings, audit, split.label_assignments
    )
    manifest = _manifest_bytes(listings, split)
    manifest_sha = hashlib.sha256(manifest).hexdigest()
    provenance = _provenance(config, manifest_sha, config.dataset.metadata_sha256)
    split_summary = {"provenance": provenance, **split.summary}
    _immutable_write(config.split.manifest_path, manifest)
    _immutable_write(config.split.summary_path, _json_bytes(split_summary))
    samples = write_reports(config, listings, audit.summary, split, provenance)
    return {
        "status": "needs_manual_review",
        "audit_status": audit.summary["status"],
        "listings": len(listings),
        "label_groups": audit.summary["label_groups"],
        "manifest": str(config.split.manifest_path),
        "manifest_sha256": manifest_sha,
        "split": split.summary,
        "inspection_samples": {key: len(value) for key, value in samples.items()},
        "report": str(config.audit.report_markdown),
    }
