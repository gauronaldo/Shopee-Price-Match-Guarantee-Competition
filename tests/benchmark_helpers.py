from __future__ import annotations

import json
import shutil
from pathlib import Path

import yaml


def make_benchmark_workspace(destination: Path, source_root: Path) -> Path:
    raw = destination / "raw"
    images = raw / "images"
    images.mkdir(parents=True)
    fixture = source_root / "tests" / "fixtures" / "smoke"
    shutil.copyfile(fixture / "train.csv", raw / "train.csv")
    for path in (fixture / "train_images").iterdir():
        shutil.copyfile(path, images / path.name)

    split_by_id = {
        "synthetic_0001": "train",
        "synthetic_0002": "train",
        "synthetic_0003": "validation",
        "synthetic_0004": "validation",
        "synthetic_0005": "test",
        "synthetic_0006": "test",
    }
    manifest = destination / "split.jsonl"
    manifest.write_text(
        "".join(
            json.dumps({"posting_id": posting_id, "split": split}, sort_keys=True) + "\n"
            for posting_id, split in split_by_id.items()
        ),
        encoding="utf-8",
    )
    config = {
        "config_version": "classical_retrieval.test.v1",
        "seed": 2026,
        "data": {
            "metadata_csv": "raw/train.csv",
            "split_manifest": "split.jsonl",
            "image_dir": "raw/images",
        },
        "evaluation": {
            "tune_split": "validation",
            "final_split": "test",
            "candidate_pool": "full_split",
            "exclude_query_itself": True,
            "recall_at": [1],
            "average_precision_at": 1,
        },
        "baselines": {
            "phash": {"candidate_k": 1},
            "tfidf": {
                "analyzer": "char_wb",
                "ngram_range": [2, 3],
                "max_features": 100,
                "candidate_k": 1,
            },
            "orb": {
                "features": 50,
                "candidate_source": "union_phash_tfidf",
                "candidate_k_per_source": 1,
                "top_k": 1,
            },
            "fusion": {"weight_grid": [0.0, 0.5, 1.0], "top_k": 1},
        },
        "artifacts": {
            "root": "artifacts/classical_retrieval",
            "report": "reports/classical_retrieval.md",
            "threshold_figure": "reports/thresholds.svg",
        },
    }
    config_path = destination / "classical_retrieval.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return config_path
