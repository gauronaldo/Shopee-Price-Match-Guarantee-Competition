# Multimodal Product Deduplication & Entity Resolution

[![Quality checks](https://img.shields.io/github/actions/workflow/status/gauronaldo/Shopee-Price-Match-Guarantee-Competition/ci.yml?branch=main&label=quality%20checks&logo=github)](https://github.com/gauronaldo/Shopee-Price-Match-Guarantee-Competition/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/Python-3.11--3.13-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.7-EE4C2C?logo=pytorch&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-4.12-5C3EE8?logo=opencv&logoColor=white)
![FAISS](https://img.shields.io/badge/FAISS-HNSW-0467DF)
![FastAPI](https://img.shields.io/badge/FastAPI-0.116-009688?logo=fastapi&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.48-FF4B4B?logo=streamlit&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?logo=docker&logoColor=white)
![Code style](https://img.shields.io/badge/code%20style-Ruff-D7FF64?logo=ruff&logoColor=261230)
![Tests](https://img.shields.io/badge/tests-102%20passing-2EA44F)

A multimodal retrieval and entity-resolution system for identifying duplicate product listings in
an e-commerce catalog. It combines visual and textual representations with candidate retrieval,
pairwise verification, and conservative graph clustering to recover product identities while
controlling false merges.

Experiments use the Kaggle **Shopee Price Match Guarantee** dataset. Competition data and trained
artifacts remain outside version control in accordance with dataset access and repository hygiene
requirements.

## Project highlights

- Leakage-safe train/validation/test splitting by product group and duplicated visual assets.
- Classical pHash, ORB, character TF-IDF, and late-fusion references alongside custom image,
  text, and multimodal PyTorch models trained from random initialization.
- Train-only hard-negative mining, FAISS HNSW candidate retrieval, calibrated pair scoring, and
  consistency-aware graph clustering.
- Frozen test evaluation across retrieval, pair classification, clustering, calibration, and
  efficiency metrics.
- A 102-test quality suite plus FastAPI, Streamlit, and Docker Compose inference paths.

## Problem context

Marketplace catalogs rarely provide a clean one-to-one mapping between listings and physical
products. Multiple sellers can describe the same item using different photos, crops, languages,
abbreviations, packaging, and promotional text. At the same time, products from one brand or
product line may appear nearly identical while differing in model number, size, quantity, color,
or flavor.

The system treats matching as a sequence of related decisions rather than a generic similarity
search:

1. retrieve a high-recall set of plausible duplicate listings;
2. verify exact-product identity using evidence from both modalities;
3. form catalog entities without allowing isolated false-positive edges to trigger large merges.

The exact matching contract and variant policy are documented in
[`docs/problem_definition.md`](docs/problem_definition.md).

## Competition and dataset

The **Shopee Price Match Guarantee** competition was hosted on Kaggle to identify listings that
refer to the same product. For each query listing, participants produced a set of matching
`posting_id` values using the product image, seller-written title, and any representations derived
from them. The challenge reflects a common catalog problem: product identity is not explicitly
shared across sellers, and neither image similarity nor title similarity is reliable on its own.

The competition task is extended into an entity-resolution system with separate candidate
retrieval, pair verification, and graph-clustering stages. This decomposition makes retrieval
misses, pair-scoring errors, and transitive cluster failures independently measurable.

The provided training metadata contains five fields:

| Field | Role |
|---|---|
| `posting_id` | Unique identifier for one seller listing |
| `image` | Filename of the associated product image |
| `image_phash` | Supplied perceptual hash used by the classical image baseline |
| `title` | Noisy, multilingual seller-written product description |
| `label_group` | Competition ground-truth product group, used only offline |

| Dataset property | Observed value |
|---|---:|
| Listings | 34,250 |
| Product groups | 11,014 |
| Unique referenced images | 32,412 |
| Median / maximum group size | 2 / 51 |
| Median image dimensions | 700 × 700 px |
| Median title length | 53 characters |

The downloadable competition test set contains only three placeholder listings because the actual
competition test labels were hidden by Kaggle. For leakage-controlled experimentation, this project
creates its own deterministic split from the labeled training release:

| Split | Listings | Product groups |
|---|---:|---:|
| Train | 27,391 | 8,817 |
| Validation | 3,430 | 1,100 |
| Test | 3,429 | 1,097 |

Splitting is performed by `label_group` and by super-components connected through exact image
references, image hashes, or perceptual hashes. As a result, the same labeled product group or
exact duplicated visual asset cannot appear in more than one split. Vocabulary construction,
checkpoint selection, thresholds, retrieval settings, and clustering rules use train and
validation only. Aggregate statistics and known label ambiguities are documented in the
[`data card`](docs/data_card.md) and
[`data quality report`](reports/data_quality_and_split.md).

## System overview

[![Multimodal product entity-resolution pipeline](assets/diagrams/shopee_entity_resolution_swiss_grid_large.drawio.svg)](assets/diagrams/shopee_entity_resolution_swiss_grid_large.drawio.svg)

The image encoder, text encoder, fusion module, losses, sampling logic, training loops, pair head,
retrieval evaluation, and clustering policy are implemented in this repository. The core neural
track is trained from random initialization; pretrained EfficientNet-B1 is evaluated later as a
separate benchmark under the same retrieval protocol.

Ground-truth `label_group` is used only for offline splitting, training, evaluation, and analysis.
It is not loaded into the demo inference contract. See
[`docs/architecture.md`](docs/architecture.md) for the training, batch, and online flows.

## Results

The held-out system evaluation was run once after checkpoints, thresholds, candidate K, and graph
rules were frozen on validation.

### Frozen test retrieval

| System or component | mAP@20 | Recall@20 | Additional result |
|---|---:|---:|---|
| Supplied pHash image baseline | 0.3073 | 0.3345 | Classical image reference |
| Custom residual image encoder | 0.5567 | 0.6594 | Random initialization |
| Custom character TextCNN | 0.7484 | 0.8698 | TF-IDF test mAP@20: 0.8564 |
| Custom multimodal pair-head rerank | 0.8685 | 0.9324 | Pair F1: 0.6843 |
| Classical late fusion | **0.8810** | **0.9349** | Pair F1: **0.7220** |
| Final-system joint retrieval | 0.8595 | 0.9324 | Recall@50: 0.9688 |

The Phase 5 pair-head row measures reranked retrieval. The final-system row reports the frozen
joint-embedding candidate stage before reciprocal-edge and clustering decisions; pair and entity
quality are reported separately below.

### Validation-only pretrained comparison

| Representation | Modalities | Initialization | mAP@20 | Recall@20 |
|---|---|---|---:|---:|
| Custom residual CNN | Image | Random | 0.5391 | 0.6467 |
| EfficientNet-B1 V2 | Image | ImageNet-1K pretrained, frozen | 0.7375 | 0.8248 |
| Custom multimodal joint embedding | Image + title | Random | **0.8702** | **0.9378** |

EfficientNet-B1 is evaluated on validation under the same exact-cosine image-retrieval protocol;
it is not fine-tuned and is not included in the frozen test comparison.

### Entity resolution

| Metric | Validation | Test |
|---|---:|---:|
| Pairwise precision | 0.9017 | 0.8959 |
| Pairwise F1 | 0.4844 | 0.4794 |
| B-cubed precision | 0.9562 | 0.9528 |
| B-cubed F1 | 0.8279 | 0.8223 |

Classical late fusion provides the strongest frozen test retrieval score. The custom multimodal
pipeline adds a trainable joint representation, hard-negative pair scoring, approximate retrieval,
and entity-level decisions. Its conservative graph policy prioritizes cluster precision and limits
transitive false merges, with lower recall on larger duplicate groups.

### Validation-only recall recovery

Follow-up validation experiments target entity fragmentation while keeping the trained encoders,
fusion module, and Phase 6 pair head frozen. Character TF-IDF statistics are fitted on train only.

| Validation policy | Candidate recall | Pair precision | Pair recall | Pair F1 | B-cubed F1 | False split |
|---|---:|---:|---:|---:|---:|---:|
| Dense Top-50 incumbent | 0.9744 | 0.9017 | 0.3312 | 0.4844 | 0.8279 | 0.3082 |
| Hybrid Top-75 + supported singleton attachment | **0.9921** | **0.8958** | **0.4557** | **0.6041** | **0.8580** | **0.2636** |

Hybrid retrieval combines frozen dense candidates, train-fitted character TF-IDF, and pHash using
weighted reciprocal-rank fusion. The result passes the predeclared validation precision, recall,
B-cubed, false-merge, and false-split gates. It has not been evaluated on the locked test split and
is presented as the candidate policy for the next frozen system version.

Full metrics, efficiency measurements, ablations, repeated seeds, and failure analyses are indexed
in [`reports/README.md`](reports/README.md). The final frozen result is in
[`reports/final_evaluation.md`](reports/final_evaluation.md).

## Demo

The demo supports:

- image-only candidate retrieval;
- title-only candidate retrieval;
- multimodal retrieval, pair scoring, and entity assignment;
- guided scenarios with three distinct curated products per scenario;
- open-ended image/title uploads;
- query-versus-candidate visual comparison and modality evidence;
- self-match exclusion for catalog-backed guided queries;
- “no confident match” and manual-review states.

Literal UTF-8 byte escapes found in some source titles, for example `\xe2\x9c\x85`, are decoded
only for presentation. The frozen text encoder still receives the original title representation
used during training.

### Local launcher

Complete [Installation and data](#installation-and-data), then restore the frozen checkpoints,
embedding caches, and entity assignments referenced by `configs/serving/demo.yaml`. Verify the
runtime contract before opening the UI:

```powershell
.venv\Scripts\python -m shopee_match.serving.cli preflight `
  --config configs\serving\demo.yaml
.venv\Scripts\python -m shopee_match.serving.cli launch
```

Open:

- Streamlit UI: `http://127.0.0.1:8501`
- FastAPI/OpenAPI: `http://127.0.0.1:8000/docs`
- Health endpoint: `http://127.0.0.1:8000/health`

Press `Ctrl+C` once in the launcher terminal to stop both services.

### Docker Compose

The API and UI are packaged with Docker Compose. The image contains the application and Python
dependencies; local `data/` and `artifacts/` are mounted read-only and are never copied into the
image.

Start Docker Desktop with the Linux container engine, then run:

```powershell
docker compose config
docker compose up --build
```

After both services become healthy, open `http://localhost:8501`. API documentation remains at
`http://localhost:8000/docs`.

```powershell
docker compose down
```

The Compose profile defaults to CPU inference. GPU containers require NVIDIA Container Toolkit
and an explicit device configuration. Runtime behavior and API contracts are documented in
[`docs/demo.md`](docs/demo.md).

## Installation and data

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install --upgrade pip
.venv\Scripts\python -m pip install -e ".[dev,eda,retrieval,pretrained,demo]"
```

Expected local-only Kaggle layout:

```text
data/raw/
  train.csv
  train_images/
  test.csv
  test_images/
  sample_submission.csv
```

The Kaggle public test directory contains only three placeholder examples. This project creates a
deterministic group-disjoint train/validation/test split from `train.csv` so that each
`label_group` occurs in exactly one split. Vocabulary, thresholds, retrieval settings, and graph
rules are selected without test leakage.

```powershell
.venv\Scripts\shopee-data prepare --config configs\data\shopee.yaml
```

The source CSV checksum and schema are verified before processing. Raw data, generated manifests,
checkpoints, indexes, caches, and detailed review artifacts are ignored by Git.

## Reproducing the pipeline

The table shows the canonical command for each system function. Completed experiment outputs are
immutable by design; use a new artifact root for a deliberate rerun instead of overwriting evidence.

| Function | Command |
|---|---|
| Smoke test | `.venv\Scripts\shopee-smoke --config configs\smoke.yaml` |
| Data audit and split | `.venv\Scripts\shopee-data prepare --config configs\data\shopee.yaml` |
| Classical baselines | `.venv\Scripts\shopee-benchmark run --config configs\experiment\classical_retrieval_benchmark.yaml` |
| Custom image training | `.venv\Scripts\shopee-image train --config configs\experiment\image_embedding_training.yaml` |
| Custom text training | `.venv\Scripts\shopee-text train --config configs\experiment\text_embedding_training.yaml` |
| Multimodal training | `.venv\Scripts\shopee-multimodal train --config configs\experiment\multimodal_embedding_training.yaml` |
| Hard-negative training | `.venv\Scripts\shopee-hard-negatives all --config configs\experiment\hard_negative_pair_head_pilot.yaml` |
| Candidate retrieval | `.venv\Scripts\shopee-retrieval benchmark --config configs\experiment\candidate_retrieval_benchmark.yaml` |
| Entity resolution | `.venv\Scripts\shopee-entity-resolution benchmark --config configs\experiment\entity_resolution_benchmark.yaml` |
| Recall-recovery graph | `.venv\Scripts\shopee-entity-resolution recover-recall --config configs\experiment\entity_recall_recovery.yaml` |
| Hybrid candidate retrieval | `.venv\Scripts\shopee-retrieval hybrid --config configs\experiment\hybrid_candidate_retrieval.yaml` |
| Hybrid entity evaluation | `.venv\Scripts\shopee-entity-resolution evaluate-hybrid-candidates --config configs\experiment\hybrid_entity_resolution.yaml` |
| Pretrained weight preparation | `.venv\Scripts\shopee-pretrained prepare-weights` |
| Pretrained comparison | `.venv\Scripts\shopee-pretrained benchmark --config configs\experiment\pretrained_image_benchmark.yaml` |
| Frozen system preflight | `.venv\Scripts\shopee-final preflight --config configs\experiment\final_system_evaluation.yaml` |

An access marker protects the single-use final test protocol. Use the preflight command to verify
the recorded evaluation rather than deleting its outputs and selecting settings from another run.

## Engineering quality

```powershell
.venv\Scripts\python -m ruff format --check .
.venv\Scripts\python -m ruff check .
.venv\Scripts\python -m mypy src\shopee_match
.venv\Scripts\python -m pytest
```

The test suite covers leakage-safe splitting, title normalization, image preprocessing, sampling,
losses, encoder shapes and gradients, exact/FAISS agreement, clustering, frozen evaluation guards,
API contracts, guided self-exclusion, and the combined launcher.

## Repository structure

```text
app/                         Streamlit demo UI
configs/                     Data, model, experiment, and serving contracts
data/                        Ignored raw/derived data and local split manifests
docs/                        Problem definition, architecture, cards, and limitations
notebooks/exploration/       Bounded EDA notebook
reports/                     Reviewed metrics and failure-analysis evidence
src/shopee_match/
  data/                      Ingestion, audit, and group-disjoint splitting
  features/ models/ losses/  Classical features and custom neural components
  training/ retrieval/       Training, mining, and candidate generation
  clustering/ evaluation/    Entity resolution and controlled evaluation
  serving/                   Frozen runtime, FastAPI, and launcher
tests/                       Synthetic fixtures, unit tests, and integration tests
```

## Documentation

- [Problem definition](docs/problem_definition.md)
- [Architecture](docs/architecture.md)
- [Data card](docs/data_card.md)
- [Model card](docs/model_card.md)
- [Error analysis](docs/error_analysis.md)
- [Demo and API](docs/demo.md)
- [Experiment reports](reports/README.md)

## Limitations

- The dataset contains noisy labels, multilingual seller text, malformed byte escapes, and
  ambiguous product variants.
- The strict clustering policy limits false merges but fragments many large product groups.
- Reported latency covers a 3,430-listing validation catalog; production-scale behavior requires
  measurement on a substantially larger index and representative request load.
- The demo searches a fixed validation catalog and omits persistent ingestion, authentication,
  rate limiting, monitoring, and artifact distribution.
- Competition data remains subject to Kaggle/Shopee access and redistribution terms.
