# Shopee Multimodal Product Matching

A reproducible research project for duplicate-product retrieval and catalog entity resolution
from product images and noisy multilingual titles. The dataset is Kaggle's **Shopee — Price
Match Guarantee** competition release.

This is not generic visual similarity: visually similar variants may be different purchasable
products, while listings with different photos and wording may refer to the same product. The
identity policy and evaluation assumptions are defined in
[`docs/problem_definition.md`](docs/problem_definition.md).

## Current status

Phases 0–2 are complete. Phase 1 validates the real
release, decodes and hashes every referenced image, audits metadata and label ambiguity, creates
a deterministic leakage-safe train/validation/test manifest, and generates an aggregate report
plus a local inspection gallery. The audit passes all critical gates but retains warnings about
cross-label duplicates and perceptually similar variants; those warnings are data properties,
not silently rewritten labels.

Phase 2 evaluates supplied-pHash, ORB, train-only character TF-IDF, and validation-tuned late
fusion under one retrieval/pair protocol. Results and failure analysis are recorded in
[`reports/classical_retrieval_benchmark.md`](reports/classical_retrieval_benchmark.md).

Phase 3 is complete. A repository-owned residual image encoder was trained from random
initialization with conservative OpenCV preprocessing, deterministic product-aware batches,
supervised contrastive loss, exact cosine evaluation, atomic checkpoints, structured diagnostics,
and concise terminal progress reporting. The frozen model reached test mAP@20 `0.55674` and
Recall@20 `0.65941`, clearly improving on supplied pHash while remaining below the
candidate-assisted ORB pipeline. See
[`reports/image_retrieval_final_comparison.md`](reports/image_retrieval_final_comparison.md).

Phase 4 is complete. The train-only, randomly initialized character TextCNN reached
validation/test mAP@20 `0.75698 / 0.74841` and Recall@20 `0.87414 / 0.86978`. It generalizes
consistently but remains below character TF-IDF test mAP@20 `0.8564`; this honest gap and the
categorized failure analysis define the role of each text signal in Phase 5. See
[`reports/text_retrieval_final_comparison.md`](reports/text_retrieval_final_comparison.md).

Phase 5 implementation, loss ablation, and full validation-only training are complete; the phase
remains open. Frozen image and
text embeddings are cached once, then a repository-owned learned fusion module and symmetric pair
head are trained without updating either encoder. Simple score fusion reached validation mAP@20
`0.87358`; the selected residual pair-head system reached `0.87903`, Recall@20 `0.93780`, and pair
F1 `0.71285`. It stopped after 7 of 30 configured epochs and retained epoch 1. Test remains locked
pending categorized failure analysis and a frozen evaluation protocol. See
[`reports/multimodal_fusion_training_summary.md`](reports/multimodal_fusion_training_summary.md).

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
.venv\Scripts\shopee-image train --config configs\experiment\image_embedding_smoke.yaml
.venv\Scripts\shopee-text train --config configs\experiment\text_embedding_smoke.yaml
```

The full image run reports training progress five times per epoch plus validation, checkpoint,
and completion stages without continuously redrawing the terminal:

```powershell
.venv\Scripts\shopee-image train --config configs\experiment\image_embedding_training.yaml
```

Use `--progress-updates-per-epoch N` to change the frequency, or set it to `0` to disable batch
progress messages. This display option does not change model training or saved experiment config.

The Phase 4 TextCNN full run uses the pilot-approved architecture and keeps test disabled:

```powershell
.venv\Scripts\shopee-text train --config configs\experiment\text_embedding_training.yaml
```

The full-history report can be regenerated from the completed local checkpoint without training
or model evaluation. Validation failures are exported for local review with separate commands:

```powershell
.venv\Scripts\shopee-text refresh-report --config configs\experiment\text_embedding_training.yaml
.venv\Scripts\shopee-text analyze-validation --config configs\experiment\text_embedding_training.yaml
```

After the text checkpoint, training configuration, training metrics, validation threshold, and
retrieval protocol are locked, the held-out test command is intentionally single-use:

```powershell
.venv\Scripts\shopee-text evaluate --config configs\experiment\text_embedding_final_evaluation.yaml
```

An existing final metrics file or report makes the evaluator refuse a second run.

Phase 5 separates the one-time frozen-embedding preparation from lightweight fusion training:

```powershell
.venv\Scripts\shopee-multimodal prepare --config configs\experiment\multimodal_embedding_smoke.yaml
.venv\Scripts\shopee-multimodal train --config configs\experiment\multimodal_embedding_smoke.yaml
.venv\Scripts\shopee-multimodal train --config configs\experiment\multimodal_embedding_pilot.yaml
.venv\Scripts\shopee-multimodal train --config configs\experiment\multimodal_residual_fusion_pilot.yaml
.venv\Scripts\shopee-multimodal train --config configs\experiment\multimodal_pair_loss_025_pilot.yaml
.venv\Scripts\shopee-multimodal train --config configs\experiment\multimodal_pair_loss_010_pilot.yaml
.venv\Scripts\shopee-multimodal train --config configs\experiment\multimodal_embedding_training.yaml
.venv\Scripts\shopee-multimodal refresh-report --config configs\experiment\multimodal_embedding_training.yaml
```

`prepare` creates or verifies train/validation caches only. Subsequent fusion runs reuse those
fingerprinted artifacts and do not recompute the frozen CNN or TextCNN embeddings.

After freezing checkpoint, training-config, training-metrics hashes, and the validation threshold,
the held-out image test evaluation is run without retraining or test-time selection:

```powershell
.venv\Scripts\shopee-image evaluate --config configs\experiment\image_embedding_final_evaluation.yaml
```

Phase 3 final results and caveats are in
[`reports/image_retrieval_final_comparison.md`](reports/image_retrieval_final_comparison.md).

For the optional EDA environment, install `-e ".[dev,eda]"` and open
`notebooks/exploration/catalog_data_exploration.ipynb`. Clear notebook outputs before every
commit.

On Linux/macOS, replace `.venv\Scripts\` with `.venv/bin/`.

The default pinned PyTorch wheel may be CPU-only. For GPU experiments, install the matching wheel
from the official PyTorch compute-platform index and verify both `torch.cuda.is_available()` and
the recorded device before starting the bounded pilot.

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
