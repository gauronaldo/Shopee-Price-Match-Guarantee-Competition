# Shopee Multimodal Product Matching

A portfolio-grade research project for duplicate-product retrieval and catalog entity resolution
using product images and noisy multilingual titles. The dataset is Kaggle's **Shopee — Price
Match Guarantee** competition data.

This is not generic image similarity: visually similar variants may be different purchasable
products, while listings with different photos and wording may refer to the same product.

## Scope and outputs

The system will support:

- **Online:** accept one image and title; return Top-K candidate `posting_id` values, calibrated
  match confidence, image/title similarity evidence, a predicted group or “no confident match,”
  and a manual-review flag.
- **Batch:** accept a catalog; return candidate pairs, pair probabilities, conservative entity
  clusters, cluster confidence, and review flags.

The exact identity policy and variant assumptions are defined in
[`docs/problem_definition.md`](docs/problem_definition.md).

## Current phase

Only **Phase 0 — problem contract and repository foundation** is implemented. It provides typed
contracts, versioned configuration, logging, seed control, a rights-safe synthetic fixture,
tests, and CI. No Kaggle data has been downloaded or inspected, and no modeling claim or metric
is reported. Phase 1 must audit the real release before baselines or neural work begins.

## Setup and checks

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -e ".[dev]"
.venv\Scripts\ruff format --check .
.venv\Scripts\ruff check .
.venv\Scripts\mypy src
.venv\Scripts\python -m pytest
.venv\Scripts\shopee-smoke --config configs\smoke.yaml
```

On Linux/macOS, replace `.venv\Scripts\` with `.venv/bin/`.

## Data access

The user must accept the Kaggle competition rules and download the dataset through their own
account. Treat it as immutable local input under `data/raw/`; never commit or redistribute it.
Kaggle credentials, raw images, derived datasets, indexes, checkpoints, and experiment outputs
are ignored by Git.

Expected training columns are `posting_id`, `image`, `image_phash`, `title`, and `label_group`.
The Phase 0 fixture mirrors these columns but contains only synthetic metadata and tiny synthetic
PPM images; it is not representative of real performance.

## Planned evidence

Later gated phases will compare pHash/ORB and TF-IDF baselines, custom image/text/multimodal
PyTorch models trained from random initialization, hard-negative mining, exact/FAISS retrieval,
pair scoring, clustering, and finally pretrained representations under the same split and
evaluation protocol. Deployment comes last.

This repository does not claim production readiness, marketplace policy compliance, or that the
competition labels are free of ambiguity.
