"""Phase 0 smoke command validating configuration, fixture, and reproducibility plumbing."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import sys
from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath

from shopee_match.config import AppConfig, load_config
from shopee_match.errors import FixtureError, ShopeeMatchError
from shopee_match.logging import configure_logging
from shopee_match.reproducibility import seed_everything

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class SmokeSummary:
    status: str
    config_version: str
    config_sha256: str
    seed: int
    records: int
    label_groups: int
    group_size_distribution: dict[int, int]
    fixture_sha256: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fixture_digest(paths: Sequence[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda item: item.as_posix()):
        digest.update(path.as_posix().encode("utf-8"))
        digest.update(bytes.fromhex(_sha256(path)))
    return digest.hexdigest()


def validate_smoke_fixture(
    config: AppConfig, project_root: Path, config_path: Path
) -> SmokeSummary:
    """Validate the tiny committed fixture without reading or creating real dataset artifacts."""
    metadata_path = project_root / config.data.metadata_csv
    image_dir = project_root / config.data.image_dir
    if not metadata_path.is_file():
        raise FixtureError(f"Fixture metadata does not exist: {config.data.metadata_csv}")
    if not image_dir.is_dir():
        raise FixtureError(f"Fixture image directory does not exist: {config.data.image_dir}")

    try:
        handle = metadata_path.open("r", encoding="utf-8", newline="")
    except OSError as exc:
        raise FixtureError(f"Cannot open fixture metadata: {exc}") from exc
    referenced_images: list[Path] = []
    posting_ids: set[str] = set()
    groups: Counter[str] = Counter()
    row_count = 0
    with handle:
        reader = csv.DictReader(handle)
        columns = set(reader.fieldnames or ())
        missing = set(config.data.required_columns) - columns
        if missing:
            raise FixtureError(f"Fixture metadata is missing columns: {sorted(missing)}")
        for row_number, row in enumerate(reader, start=2):
            row_count += 1
            posting_id = (row.get("posting_id") or "").strip()
            label_group = (row.get("label_group") or "").strip()
            title = (row.get("title") or "").strip()
            image_name = (row.get("image") or "").strip()
            image_phash = (row.get("image_phash") or "").strip()
            if not all((posting_id, label_group, title, image_name, image_phash)):
                raise FixtureError(f"Empty required value at fixture row {row_number}")
            if posting_id in posting_ids:
                raise FixtureError(
                    f"Duplicate posting_id at fixture row {row_number}: {posting_id}"
                )
            image_path_variants = (PurePosixPath(image_name), PureWindowsPath(image_name))
            if any(path.is_absolute() or len(path.parts) != 1 for path in image_path_variants):
                raise FixtureError(
                    f"Unsafe image filename at fixture row {row_number}: {image_name}"
                )
            image_path = image_dir / image_name
            if not image_path.is_file():
                raise FixtureError(f"Missing fixture image at row {row_number}: {image_name}")
            posting_ids.add(posting_id)
            groups[label_group] += 1
            referenced_images.append(image_path)
    if not row_count:
        raise FixtureError("Fixture metadata contains no rows")
    size_distribution = Counter(groups.values())
    state = seed_everything(config.project.seed)
    return SmokeSummary(
        status="ok",
        config_version=config.config_version,
        config_sha256=_sha256(config_path),
        seed=state.seed,
        records=row_count,
        label_groups=len(groups),
        group_size_distribution=dict(sorted(size_distribution.items())),
        fixture_sha256=_fixture_digest([metadata_path, *referenced_images]),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="shopee-smoke")
    parser.add_argument("--config", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config = load_config(args.config)
        configure_logging(config.logging.level)
        summary = validate_smoke_fixture(config, Path.cwd(), args.config)
    except ShopeeMatchError as exc:
        configure_logging()
        LOGGER.error("%s", exc)
        return 2
    sys.stdout.write(json.dumps(asdict(summary), sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
