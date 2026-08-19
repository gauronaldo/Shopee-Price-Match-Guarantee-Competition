# Shopee Multimodal Product Matching

A reproducible research project for duplicate-product retrieval and catalog entity resolution
from product images and noisy multilingual titles. The dataset is Kaggle's **Shopee — Price
Match Guarantee** competition release.

This is not generic visual similarity: visually similar variants may be different purchasable
products, while listings with different photos and wording may refer to the same product. The
identity policy and evaluation assumptions are defined in
[`docs/problem_definition.md`](docs/problem_definition.md).

## Current status

Phases 0–2 are implemented. Phase 1 validates the real
release, decodes and hashes every referenced image, audits metadata and label ambiguity, creates
a deterministic leakage-safe train/validation/test manifest, and generates an aggregate report
plus a local inspection gallery. The audit passes all critical gates but retains warnings about
cross-label duplicates and perceptually similar variants; those warnings are data properties,
not silently rewritten labels.

Phase 2 evaluates supplied-pHash, ORB, train-only character TF-IDF, and validation-tuned late
fusion under one retrieval/pair protocol. Results and failure analysis are recorded in
[`reports/classical_retrieval_benchmark.md`](reports/classical_retrieval_benchmark.md). No neural
model has been trained;
Phase 3 is the next phase and must begin with the scratch image-encoder smoke gates.

## Setup, checks, and data preparation

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -e ".[dev]"
.venv\Scripts\ruff format --check .
.venv\Scripts\ruff check .
.venv\Scripts\mypy src
.venv\Scripts\python -m pytest
.venv\Scripts\shopee-smoke --config configs\smoke.yaml
.venv\Scripts\shopee-data prepare --config configs\data\shopee.yaml
.venv\Scripts\shopee-benchmark run --config configs\experiment\classical_retrieval_benchmark.yaml
```

For the optional EDA environment, install `-e ".[dev,eda]"` and open
`notebooks/exploration/catalog_data_exploration.ipynb`. Clear notebook outputs before every
commit.

On Linux/macOS, replace `.venv\Scripts\` with `.venv/bin/`.

The authorized Kaggle release is expected in this local-only layout:

```text
data/raw/
  train.csv
  train_images/
  test.csv
  test_images/
  sample_submission.csv
```

Raw images/CSVs, generated split manifests, full audit JSON, inspection galleries, caches,
checkpoints, and run artifacts are ignored by Git. The aggregate audit and non-sensitive SVG
figures are kept under `reports/`. Phase 1 refuses an unexpected CSV checksum or schema, missing
or corrupt images, duplicate posting IDs, and conflicting immutable outputs.

## Repository layout

```text
configs/                         versioned data, model, and experiment inputs
data/                            ignored raw/derived data and frozen local split manifests
docs/                            product contract, data/model cards, error analysis
notebooks/exploration/           bounded diagnostics only
reports/                         reviewed aggregate reports and lightweight figures
scripts/                         thin command adapters only
src/shopee_match/
  data/                          Phase 1 ingestion, audit, split, reporting
  features/ models/ losses/      later modeling components
  training/ retrieval/           training and candidate retrieval
  clustering/ evaluation/        entity resolution and controlled evaluation
  serving/                       inactive until the deployment phase
tests/                           synthetic fixtures, unit and integration tests
app/                             inactive until the final application phase
```

This root file is the single repository README; individual folders intentionally do not contain
separate README files.

## Phase 1 outputs

- Aggregate report: [`reports/data_audit_v1.md`](reports/data_audit_v1.md)
- Data card: [`docs/data_card.md`](docs/data_card.md)
- Local manifest: `data/splits/shopee_group_split_v1.jsonl` (ignored)
- Local manifest summary: `data/splits/shopee_group_split_v1.summary.json` (ignored)
- Local pair gallery: `reports/figures/generated/data_audit_v1/gallery.html` (ignored)

The split unit is a leakage super-component: label groups connected by an exact image reference,
image SHA-256, or exact pHash are kept together. Near-pHash matches are audited but not merged
automatically because low Hamming distance can represent a legitimate size, volume, color, or
packaging variant.

## Scope

The eventual online interface returns Top-K candidate posting IDs, calibrated match confidence,
image/title evidence, a predicted group or “no confident match,” and a review flag. Batch mode
returns candidate pairs, pair probabilities, conservative clusters, and review flags.

The project does not claim production readiness, marketplace-policy compliance, or that the
competition labels are perfect commercial identity ground truth. Raw competition content is not
redistributed.
