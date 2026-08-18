from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

import yaml


def make_dataset_workspace(destination: Path, source_root: Path) -> Path:
    raw = destination / "raw"
    images = raw / "images"
    images.mkdir(parents=True)
    metadata_source = source_root / "tests" / "fixtures" / "smoke" / "train.csv"
    metadata = raw / "train.csv"
    shutil.copyfile(metadata_source, metadata)
    for path in (source_root / "tests" / "fixtures" / "smoke" / "train_images").iterdir():
        shutil.copyfile(path, images / path.name)
    metadata_sha = hashlib.sha256(metadata.read_bytes()).hexdigest()
    config = {
        "config_version": "phase1.test.v1",
        "dataset": {
            "name": "synthetic_shopee_fixture",
            "competition_slug": "synthetic",
            "metadata_csv": "raw/train.csv",
            "metadata_sha256": metadata_sha,
            "image_dir": "raw/images",
            "required_columns": [
                "posting_id",
                "image",
                "image_phash",
                "title",
                "label_group",
            ],
        },
        "split": {
            "strategy_version": "leakage_super_component.v1",
            "manifest_path": "outputs/split.jsonl",
            "summary_path": "outputs/split.summary.json",
            "seed": 2026,
            "train_fraction": 0.8,
            "validation_fraction": 0.1,
            "test_fraction": 0.1,
            "group_key": "label_group",
            "link_exact_image_reference": True,
            "link_exact_sha256": True,
            "link_exact_phash": True,
        },
        "audit": {
            "report_json": "outputs/audit.json",
            "report_markdown": "outputs/audit.md",
            "figure_dir": "outputs/figures",
            "inspection_dir": "outputs/inspection",
            "random_sample_seed": 2026,
            "same_group_samples": 3,
            "different_group_samples": 3,
            "near_phash_hamming_distance": 4,
        },
    }
    config_path = destination / "phase1.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return config_path
