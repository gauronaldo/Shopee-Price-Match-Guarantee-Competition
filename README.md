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
[`reports/classical_retrieval.md`](reports/classical_retrieval.md).

Phase 3 is complete. A repository-owned residual image encoder was trained from random
initialization with conservative OpenCV preprocessing, deterministic product-aware batches,
supervised contrastive loss, exact cosine evaluation, atomic checkpoints, structured diagnostics,
and concise terminal progress reporting. The frozen model reached test mAP@20 `0.55674` and
Recall@20 `0.65941`, clearly improving on supplied pHash while remaining below the
candidate-assisted ORB pipeline. See
[`reports/image_encoder.md`](reports/image_encoder.md).

Phase 4 is complete. The train-only, randomly initialized character TextCNN reached
validation/test mAP@20 `0.75698 / 0.74841` and Recall@20 `0.87414 / 0.86978`. It generalizes
consistently but remains below character TF-IDF test mAP@20 `0.8564`; this honest gap and the
categorized failure analysis define the role of each text signal in Phase 5. See
[`reports/text_encoder.md`](reports/text_encoder.md).

Phase 5 is complete. Frozen image and text embeddings are cached once, then a repository-owned
residual fusion module and symmetric pair head are trained without updating either encoder. Across
three seeds, validation mAP@20 is `0.88008 ± 0.00132`. The canonical frozen-test result is mAP@20
`0.86848`, Recall@20 `0.93235`, and pair F1 `0.68429`. It beats custom unimodal systems and TF-IDF
retrieval on mAP, but remains below the strongest classical fused baseline (`0.8810` test mAP@20).
See [`reports/multimodal_model.md`](reports/multimodal_model.md).

Phase 6 is complete on validation. Exact train-only mining produced `24,332` deterministic
cross-label hard-negative pairs after pHash/title false-negative guards and a `50%` variant cap.
The initial joint fine-tuning pilot regressed and was rejected. Freezing fusion and updating only
the pair head passed all three seeds: mean validation mAP@20 is `0.87926` (population std.
`0.00007`), controlled-recall precision improves by `+0.00244` on average, and Recall@20 is
unchanged. The improvement is real but modest; Phase 6 did not access test. See
[`reports/hard_negative_mining.md`](reports/hard_negative_mining.md).

Phase 7 is complete on validation. Deterministic exact cosine search selected `K=50` as the
smallest tested candidate budget that exceeds the `0.95` macro-recall target, reaching
Recall@50 `0.97438`. FAISS HNSW with `efSearch=64` preserves the same Recall@50 and has `0.99851`
candidate-set agreement with exact search, while reducing measured single-query p50 latency from
`0.414 ms` to `0.200 ms`. Test remains untouched. See
[`reports/candidate_retrieval.md`](reports/candidate_retrieval.md).

Phase 8 is complete on validation. The frozen Phase 6 pair head scores the Phase 7 Top-50
candidates, then reciprocal-neighbour edges and full cross-component support create conservative
entities without using labels as graph features. The selected policy reaches pairwise precision
`0.90165`, pairwise F1 `0.48444`, and B-cubed F1 `0.82794`; false-split group rate remains
`0.30818`, especially for large diverse groups. Test remains untouched. See
[`reports/entity_resolution.md`](reports/entity_resolution.md).

Phase 9 is complete on validation. The SHA-256-verified TorchVision EfficientNet-B1
`IMAGENET1K_V2` representation reaches mAP@20 `0.73753` and Recall@20 `0.82481`. It clearly
beats the custom image-only encoder trained from random initialization (`0.53907 / 0.64667`) but
remains below the custom multimodal joint representation (`0.87023 / 0.93780`). This is a frozen
comparison with no local fine-tuning and no test access. See
[`reports/pretrained_benchmark.md`](reports/pretrained_benchmark.md).

## Setup, checks, and data preparation

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -e ".[dev,retrieval]"
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
.venv\Scripts\shopee-multimodal analyze-validation --config configs\experiment\multimodal_embedding_training.yaml
.venv\Scripts\shopee-multimodal train --config configs\experiment\multimodal_embedding_seed_2027.yaml
.venv\Scripts\shopee-multimodal train --config configs\experiment\multimodal_embedding_seed_2028.yaml
.venv\Scripts\shopee-multimodal train --config configs\experiment\multimodal_contrastive_only_ablation.yaml
.venv\Scripts\shopee-multimodal train --config configs\experiment\multimodal_pair_only_ablation.yaml
.venv\Scripts\shopee-multimodal evaluate-frozen-test --config configs\experiment\multimodal_embedding_final_evaluation.yaml
```

`prepare` creates or verifies train/validation caches only. Subsequent fusion runs reuse those
fingerprinted artifacts and do not recompute the frozen CNN or TextCNN embeddings.
The frozen-test command is one-time by design and refuses to overwrite an existing metrics file or
report.

Phase 6 separates immutable mining evidence from fine-tuning. `mine` uses train only, `train` uses
train plus validation, and `summarize` verifies frozen hashes across the three seeds. The canonical
manual sequence is:

```powershell
.venv\Scripts\shopee-hard-negatives mine --config configs\experiment\hard_negative_pair_head_pilot.yaml
.venv\Scripts\shopee-hard-negatives train --config configs\experiment\hard_negative_pair_head_pilot.yaml
.venv\Scripts\shopee-hard-negatives all --config configs\experiment\hard_negative_pair_head_seed_2027.yaml --progress-updates-per-epoch 2
.venv\Scripts\shopee-hard-negatives all --config configs\experiment\hard_negative_pair_head_seed_2028.yaml --progress-updates-per-epoch 2
.venv\Scripts\shopee-hard-negatives summarize --config configs\experiment\hard_negative_repeated_seed_summary.yaml
```

Each versioned output refuses silent overwrite. To repeat a completed experiment, choose a new
artifact root/report path or deliberately remove only that local ignored run after reviewing it.

Phase 7 extracts one frozen joint embedding per validation listing, verifies exact search first,
then benchmarks FAISS HNSW and selects `K`/`efSearch` without accessing test:

```powershell
.venv\Scripts\shopee-retrieval benchmark --config configs\experiment\candidate_retrieval_benchmark.yaml
```

The command refuses to overwrite completed metrics or the tracked report. Use a distinct artifact
root and report path for a deliberate rerun.

Phase 8 reuses the frozen validation embeddings and pair head, scores each undirected Top-50 pair
once, and selects a conservative reciprocal-neighbour clustering policy:

```powershell
.venv\Scripts\shopee-entity-resolution benchmark `
  --config configs\experiment\entity_resolution_benchmark.yaml
```

The command writes ignored scored-pair, assignment, metrics, and failure-review artifacts plus the
reviewed aggregate report. It refuses to overwrite an existing completed Phase 8 run.

Phase 9 installs the optional TorchVision dependency, acquires the official weight through its
weight enum, verifies the full SHA-256, and evaluates the frozen image representation:

```powershell
.venv\Scripts\python -m pip install -e ".[dev,retrieval,pretrained]"
.venv\Scripts\shopee-pretrained prepare-weights
.venv\Scripts\shopee-pretrained benchmark `
  --config configs\experiment\pretrained_image_benchmark.yaml
```

The benchmark uses the same validation split and exact Top-50 protocol as Phase 7. It does not
fine-tune EfficientNet or access test, and it refuses to overwrite completed evidence.

After freezing checkpoint, training-config, training-metrics hashes, and the validation threshold,
the held-out image test evaluation is run without retraining or test-time selection:

```powershell
.venv\Scripts\shopee-image evaluate --config configs\experiment\image_embedding_final_evaluation.yaml
```

Phase 3 final results and caveats are in
[`reports/image_encoder.md`](reports/image_encoder.md).

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
reports/                         function-grouped reports, index, and lightweight figures
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

This root file is the project entry point. `reports/README.md` is the only secondary README and acts
as a navigation index for function-grouped experiment evidence.

## Phase 1 outputs

- Aggregate report: [`reports/data_quality_and_split.md`](reports/data_quality_and_split.md)
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
