"""Immutable source ingestion and OpenCV image validation."""

from __future__ import annotations

import csv
import hashlib
import re
from collections import Counter
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

import cv2
import numpy as np

from shopee_match.data.config import Phase1Config
from shopee_match.data.models import Finding, ImageRecord, Listing
from shopee_match.errors import DataValidationError

PHASH_PATTERN = re.compile(r"^[0-9a-fA-F]{16}$")


def sha256_file(path: Path) -> str:
    """Return a streaming SHA-256 for an immutable source or generated artifact."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_listings(config: Phase1Config) -> tuple[list[Listing], list[Finding]]:
    """Read the source CSV strictly, preserving raw title text."""
    path = config.dataset.metadata_csv
    if not path.is_file():
        raise DataValidationError(f"Metadata CSV does not exist: {path}")
    actual_sha = sha256_file(path)
    if actual_sha != config.dataset.metadata_sha256:
        raise DataValidationError(
            f"Metadata checksum mismatch for {path}: expected "
            f"{config.dataset.metadata_sha256}, got {actual_sha}"
        )
    try:
        handle = path.open("r", encoding="utf-8", newline="")
    except OSError as exc:
        raise DataValidationError(f"Cannot open metadata CSV {path}: {exc}") from exc
    findings: list[Finding] = []
    listings: list[Listing] = []
    with handle:
        reader = csv.DictReader(handle)
        columns = set(reader.fieldnames or ())
        missing_columns = set(config.dataset.required_columns) - columns
        if missing_columns:
            raise DataValidationError(f"Missing required columns: {sorted(missing_columns)}")
        for row_number, row in enumerate(reader, start=2):
            values = {key: (row.get(key) or "").strip() for key in config.dataset.required_columns}
            missing_values = [key for key, value in values.items() if not value]
            if missing_values:
                findings.append(
                    Finding(
                        "critical",
                        "missing_required_value",
                        f"Row {row_number} is missing {missing_values}",
                        1,
                    )
                )
                continue
            image_variants = (
                PurePosixPath(values["image"]),
                PureWindowsPath(values["image"]),
            )
            if any(item.is_absolute() or len(item.parts) != 1 for item in image_variants):
                findings.append(
                    Finding(
                        "critical",
                        "unsafe_image_path",
                        f"Row {row_number} has unsafe image path {values['image']!r}",
                        1,
                    )
                )
                continue
            if not PHASH_PATTERN.fullmatch(values["image_phash"]):
                findings.append(
                    Finding(
                        "critical",
                        "invalid_phash",
                        f"Row {row_number} has invalid 64-bit pHash",
                        1,
                    )
                )
                continue
            listings.append(
                Listing(
                    row_number,
                    values["posting_id"],
                    values["image"],
                    values["image_phash"].lower(),
                    values["title"],
                    values["label_group"],
                )
            )
    posting_counts = Counter(item.posting_id for item in listings)
    duplicates = sum(count - 1 for count in posting_counts.values() if count > 1)
    if duplicates:
        findings.append(
            Finding("critical", "duplicate_posting_id", "Duplicate posting IDs", duplicates)
        )
    row_counts = Counter(
        (item.posting_id, item.image, item.image_phash, item.title, item.label_group)
        for item in listings
    )
    duplicate_rows = sum(count - 1 for count in row_counts.values() if count > 1)
    if duplicate_rows:
        findings.append(Finding("critical", "duplicate_row", "Duplicate full rows", duplicate_rows))
    return listings, findings


def audit_images(
    listings: list[Listing], image_dir: Path
) -> tuple[dict[str, ImageRecord], list[Finding], dict[str, Any]]:
    """Validate all referenced images using OpenCV and hash their immutable bytes."""
    if not image_dir.is_dir():
        raise DataValidationError(f"Image directory does not exist: {image_dir}")
    referenced = {item.image for item in listings}
    available = {path.name for path in image_dir.iterdir() if path.is_file()}
    missing = sorted(referenced - available)
    findings: list[Finding] = []
    if missing:
        findings.append(
            Finding(
                "critical",
                "missing_image",
                f"Referenced images are missing; first examples: {missing[:5]}",
                len(missing),
            )
        )
    unreferenced = available - referenced
    if unreferenced:
        findings.append(
            Finding(
                "warning",
                "unreferenced_image",
                "Files exist in the train image directory but are not referenced",
                len(unreferenced),
            )
        )
    records: dict[str, ImageRecord] = {}
    decode_failures: list[str] = []
    for image_name in sorted(referenced & available):
        path = image_dir / image_name
        raw = path.read_bytes()
        decoded = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
        if decoded is None:
            decode_failures.append(image_name)
            continue
        height, width = decoded.shape[:2]
        records[image_name] = ImageRecord(
            image_name,
            path,
            hashlib.sha256(raw).hexdigest(),
            int(width),
            int(height),
            len(raw),
        )
    if decode_failures:
        findings.append(
            Finding(
                "critical",
                "image_decode_failure",
                f"OpenCV could not decode images; first examples: {decode_failures[:5]}",
                len(decode_failures),
            )
        )
    stats: dict[str, Any] = {
        "referenced": len(referenced),
        "available": len(available),
        "decoded": len(records),
        "missing": len(missing),
        "unreferenced": len(unreferenced),
        "decode_failures": len(decode_failures),
        "opencv_version": cv2.__version__,
    }
    return records, findings, stats
