# End-to-End Pipeline Guide

## 1. Purpose and current boundary

This document explains the complete system implemented through Phase 7 of the project. It follows
one product listing from raw Kaggle data to a verified retrieval candidate set and explains why
each stage exists, what it learns, what it produces, and how it is evaluated.

The implemented boundary is deliberately explicit:

- Phases 0–7 are complete.
- Phase 7 returns high-recall candidate listings; it does not make the final duplicate decision.
- Pair calibration, graph construction, and entity-resolution clustering belong to Phase 8 and
  are not yet implemented.
- Pretrained-model comparisons belong to Phase 9. The core learned system described here is
  trained from random initialization without external pretrained weights.

## 2. Problem in plain language

A **listing** is one seller-facing catalog entry. It contains a unique `posting_id`, one image,
a title, and a supplied perceptual image hash. Several listings may refer to the same underlying
product even when sellers use different photographs, languages, abbreviations, or formatting.

During training, Kaggle provides `label_group`. All listings with the same `label_group` are
treated as the same product identity. This field is ground truth for model development; it would
normally be unavailable for new production listings. A deployed system must infer likely matches
from image and title evidence instead.

The practical workflow therefore has two separate questions:

1. **Candidate retrieval:** can the system quickly find nearly all plausible duplicates?
2. **Final decision and grouping:** which retrieved pairs are true matches, and which listings
   should be placed in the same product entity?

Keeping these questions separate is important. A retrieval system should favor recall because a
missed candidate cannot be recovered later. The downstream pair scorer and clustering rules can
then be more conservative about precision.

## 3. System map

```text
Kaggle metadata + images
          |
          v
Schema/image audit -----> deterministic label-group-disjoint split
          |                         |
          |                         +--> train: fit every learnable statistic
          |                         +--> validation: select models, K, thresholds
          |                         +--> test: one-time frozen evaluations only
          v
Classical reference systems
  pHash | ORB | character TF-IDF | score fusion
          |
          v
Scratch image encoder + scratch text encoder
          |                         |
          +------ frozen embeddings+
                         |
                         v
             learned multimodal fusion
                         |
                         +--> joint listing embedding
                         +--> symmetric pair head
                         |
                         v
                 hard-negative mining
                         |
                         v
       exact cosine reference + FAISS HNSW benchmark
                         |
                         v
              Top-50 candidate posting IDs
                         |
                         v
        Phase 8 pair scoring and clustering (next)
```

## 4. Phase 0 — Contract and repository foundation

Phase 0 turns an open-ended competition problem into a testable engineering contract.

### What was defined

- “Same product” means the same purchasable identity, not merely visual similarity.
- Quantity, size, model number, color, and bundle differences may indicate different products.
- Online output will eventually contain Top-K candidates, confidence, modality evidence, and a
  review state.
- Batch output will eventually contain candidate pairs, pair probabilities, entity identifiers,
  cluster confidence, and review flags.

### Engineering controls

- Experiments are configured through YAML rather than hidden notebook state.
- Random seeds, device, environment, Git state, and artifact hashes are recorded.
- Tests use synthetic fixtures and do not require the private competition data.
- Raw data, checkpoints, embedding caches, indexes, and temporary diagnostics are excluded from
  Git.
- Reusable logic lives under `src/shopee_match/`; notebooks are limited to exploration.

These controls do not improve a metric directly. They make later results explainable and
repeatable.

## 5. Phase 1 — Data audit and leakage-safe splitting

### Data checks

The ingestion pipeline validates the expected columns, unique posting IDs, referenced image
files, OpenCV decodability, duplicate rows, missing values, image dimensions, title statistics,
group sizes, scripts, pHash collisions, and suspicious cross-label similarities.

OpenCV is used here as a data-quality tool. A filename existing on disk is insufficient: the
image must decode into a valid pixel array with acceptable dimensions and channels.

### Why the split is made by group

Rows are never split independently. Every `label_group` is assigned wholly to train,
validation, or test. Otherwise, one photograph or near-identical title from a product group could
appear during training while another listing from the same group appears during evaluation. That
would measure memorization rather than generalization to unseen products.

The project also audits near-identical files and pHash relationships across proposed splits. The
result is a deterministic manifest whose integrity tests assert that group identities do not
cross split boundaries.

### Split responsibilities

| Split | Allowed use |
|---|---|
| Train | Fit vocabularies, IDF, model weights, hard-negative manifests, and normalization data |
| Validation | Choose checkpoints, fusion settings, thresholds, candidate K, and ANN parameters |
| Test | One-time evaluation after checkpoint, protocol, and selection rules are frozen |

Detailed evidence is in [the data-quality report](../reports/data_quality_and_split.md) and
[the data card](data_card.md).

## 6. Phase 2 — Classical reference systems

Classical systems establish strong, inexpensive reference points before neural training.

### Supplied pHash

A perceptual hash compresses the coarse visual structure of an image into a short bit pattern.
Hamming distance counts how many bits differ. It is fast and useful for identical or lightly
edited images, but it cannot reliably understand different views of the same product.

### ORB local features

ORB detects local keypoints such as corners, creates binary descriptors around them, and matches
descriptors between images. It can recognize overlapping local regions after moderate crop,
rotation, or overlay changes. It is more expensive than pHash and can be confused by repeated
packaging patterns. The measured ORB result is candidate-assisted, so its candidate source is
part of the reported method rather than hidden preprocessing.

### Character n-gram TF-IDF

Titles are normalized while preserving identity-critical digits and units. Character fragments
receive high weight when they are distinctive in the training corpus and low weight when they are
common. Cosine similarity then compares sparse title vectors. Character fragments tolerate noisy
spacing, misspellings, abbreviations, and multilingual titles better than strict word matching.

Vocabulary and IDF statistics are fit on train only.

### Late score fusion

The image and text similarities are calibrated on validation and combined as scores. This is
called late fusion because the two representations remain independent until the final score.
It is simple, interpretable, and remains the strongest held-out classical reference in this
project.

| Held-out test method | mAP@20 | Recall@20 | Pair F1 |
|---|---:|---:|---:|
| Supplied pHash | 0.3073 | 0.3345 | 0.3607 |
| Candidate-assisted ORB | 0.6577 | 0.8151 | 0.5766 |
| Character TF-IDF | 0.8564 | 0.9291 | 0.7048 |
| Classical late fusion | **0.8810** | **0.9349** | **0.7220** |

The full protocol, thresholds, latency, and failure cases are in
[the classical retrieval report](../reports/classical_retrieval.md).

## 7. Phase 3 — Scratch image encoder

### Architecture

The image model is a small residual CNN implemented in PyTorch. Convolutional layers learn local
visual patterns; residual connections let deeper blocks learn refinements without discarding the
original signal. Global average pooling converts spatial feature maps into one vector, and a
projection head produces an L2-normalized image embedding.

All core weights start randomly. No ImageNet or other pretrained backbone is loaded.

### Training method

- Conservative resize, crop, color, compression, noise, and rotation augmentations expose the
  model to realistic listing variation without intentionally erasing variant evidence.
- Product-aware `P × K` batches sample several product groups and several listings per group.
- Supervised contrastive loss pulls same-group embeddings together while pushing other groups
  apart inside the batch.
- Exact cosine retrieval on validation selects the checkpoint.

L2 normalization makes dot product equal cosine similarity, so training and retrieval use one
consistent geometric interpretation.

### Result

The frozen image model reaches test mAP@20 `0.55674` and Recall@20 `0.65941`. It clearly beats
pHash because it can learn product-level visual patterns, but it remains below the
candidate-assisted ORB result. That is plausible for a compact model trained from random
initialization and is documented rather than hidden.

See [the image encoder report](../reports/image_encoder.md).

## 8. Phase 4 — Scratch text encoder

### Text representation

The normalization pipeline retains digits and measurement units because `128 GB`, `500 ml`, and
`2 pcs` can distinguish otherwise similar products. A character vocabulary is built only from
train. Each character is mapped to a randomly initialized embedding.

### TextCNN

Parallel one-dimensional convolutions learn short character patterns of different lengths.
Pooling keeps the strongest evidence from each filter, and a projection head produces a
normalized title embedding. This design is compact and robust to spelling noise, unusual word
boundaries, and multiple languages.

The same product-aware batching and supervised contrastive objective used for images are adapted
to titles.

### Result

The text encoder reaches test mAP@20 `0.74841` and Recall@20 `0.86978`. It generalizes
consistently from validation but remains below TF-IDF. This result shows that a more complex model
is not automatically better: TF-IDF is exceptionally strong for rare product codes and exact
character fragments.

See [the text encoder report](../reports/text_encoder.md).

## 9. Phase 5 — Scratch multimodal model

### Frozen unimodal inputs

The best image and text encoders are loaded and used once to create fingerprinted embedding
caches. Their weights remain frozen. Fusion experiments can therefore run quickly without
re-extracting images or accidentally changing the Phase 3 and Phase 4 references.

These encoders are still part of the scratch track: both were trained locally from random
initialization.

### Learned fusion

For an image embedding `v` and text embedding `t`, the fusion module uses the modalities together
and learns a residual correction to a stable simple-fusion reference. The output is a normalized
joint listing embedding `z`.

The residual design gives the model a safe starting point. It can keep a useful average-like
combination while learning when image or text should contribute differently.

### Symmetric pair head

For two listings, the pair head uses symmetric evidence such as:

```text
z1 * z2        element-wise agreement
|z1 - z2|     element-wise disagreement
```

Swapping the listings does not change these features. Binary cross-entropy trains the head to
separate matching and non-matching pairs. Supervised contrastive loss simultaneously maintains a
retrieval-friendly joint embedding space.

### Ablation discipline

The project compares image only, text only, simple fusion, learned fusion, pair-head reranking,
contrastive-only, pair-only, and multiple loss weights. It also repeats the selected validation
configuration across three seeds before the frozen test is accessed.

### Result

The selected model reaches test mAP@20 `0.86848`, Recall@20 `0.93235`, and pair F1 `0.68429`.
It strongly improves over either scratch encoder alone. It slightly trails classical late fusion
on held-out mAP and pair F1, which is an honest and useful result: learned multimodal features add
a reusable dense representation, but exact character evidence remains unusually competitive.

See [the multimodal report](../reports/multimodal_model.md).

## 10. Phase 6 — Hard-negative mining

### What is a hard negative?

A hard negative is a different product that the current model ranks as very similar. Typical
examples share a brand and package design but differ in volume, color, model number, or bundle
quantity. Random negatives are often too easy and provide little learning signal.

### Mining process

1. Extract current joint embeddings for train listings only.
2. Run deterministic exact nearest-neighbor search.
3. Keep high-scoring cross-label pairs.
4. Apply pHash/title guards to reduce suspected false negatives caused by questionable labels.
5. Cap variant categories so one failure type cannot dominate training.
6. Save the manifest together with source checkpoint and configuration fingerprints.

This produced `24,332` deterministic training pairs.

### Controlled experiment

Joint fine-tuning initially regressed and was rejected. The accepted experiment freezes the
fusion representation and updates only the pair head with a controlled mixture of random and hard
negatives. Across three seeds it preserves Recall@20 and improves precision at the fixed recall
target by `+0.00244` on average. The gain is small but consistent; Phase 6 does not access test.

See [the hard-negative report](../reports/hard_negative_mining.md).

## 11. Phase 7 — Candidate generation

Phase 7 converts the learned joint representation into a practical search component.

### Exact search as the reference

Every validation listing has one normalized joint embedding. Exact cosine search compares each
query with every reference embedding and provides the ground-truth implementation for system
testing. Candidate budgets are evaluated on validation; `K=50` is the smallest tested value that
exceeds the configured `0.95` macro Recall@K target.

### FAISS HNSW

HNSW builds a layered proximity graph. A query traverses the graph instead of comparing against
every vector. `efSearch` controls how broadly the graph is explored: larger values usually improve
agreement with exact search but cost more latency.

Approximate search is accepted only after comparison with exact search on the same embeddings.
At `efSearch=64`, measured Recall@50 is unchanged and candidate-set agreement with exact search
is `0.99851`.

### Latest validation result

| Metric | Exact cosine | FAISS HNSW |
|---|---:|---:|
| Recall@50 | 0.97438 | 0.97438 |
| mAP@50 | 0.87279 | 0.87279 |
| Hit@50 | 0.98863 | 0.98863 |
| Single-query p50 latency | 0.414 ms | 0.200 ms |
| Single-query p95 latency | 0.573 ms | 0.297 ms |
| Batch throughput | 4,067.76 q/s | 7,024.92 q/s |

Latency is hardware- and implementation-dependent. It is evidence for this run, not a universal
production guarantee. Test remains untouched because K and ANN settings are still
validation-selected system parameters.

See [the candidate retrieval report](../reports/candidate_retrieval.md).

## 12. How to read the metrics

No single number describes the entire system.

| Metric | Question answered | Primary use |
|---|---|---|
| Recall@K | What fraction of each query's true partners appear in Top-K? | Candidate coverage |
| mAP@K | Are true partners retrieved early and consistently? | Retrieval ranking quality |
| Hit@K | Does a query retrieve at least one true partner? | Basic discovery coverage |
| Pair precision | Of accepted pairs, how many are correct? | False-match control |
| Pair recall | Of true pairs, how many are accepted? | Miss control |
| Pair F1 | What is the precision/recall balance at one threshold? | Pair decision summary |
| p50/p95 latency | What do typical and slower queries cost? | Operational behavior |
| Candidate agreement | How closely does ANN reproduce exact Top-K sets? | ANN correctness check |

The current most objective headline depends on the component being reviewed:

- use held-out test mAP@20 for final ranking comparisons through Phase 5;
- use validation Recall@50 for the Phase 7 candidate-generation gate;
- do not compare validation and test numbers as if they came from one protocol;
- do not treat global pair F1 as the Kaggle competition's mean per-query F1.

## 13. Reproducible manual workflow

### Environment and checks

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -e ".[dev,retrieval]"
.venv\Scripts\ruff format --check .
.venv\Scripts\ruff check .
.venv\Scripts\mypy src
.venv\Scripts\python -m pytest
.venv\Scripts\shopee-smoke --config configs\smoke.yaml
```

### Data and classical baseline

```powershell
.venv\Scripts\shopee-data prepare --config configs\data\shopee.yaml
.venv\Scripts\shopee-benchmark run --config configs\experiment\classical_retrieval_benchmark.yaml
```

### Scratch image and text training

```powershell
.venv\Scripts\shopee-image train --config configs\experiment\image_embedding_training.yaml
.venv\Scripts\shopee-text train --config configs\experiment\text_embedding_training.yaml
```

Their final evaluation configurations verify frozen SHA-256 fingerprints and refuse unintended
overwrite. The test commands should not be rerun merely to reproduce a console display; the
tracked reports already contain the frozen results.

### Multimodal preparation and training

```powershell
.venv\Scripts\shopee-multimodal prepare --config configs\experiment\multimodal_embedding_training.yaml
.venv\Scripts\shopee-multimodal train --config configs\experiment\multimodal_embedding_training.yaml
```

`prepare` extracts and fingerprints the frozen image/text embeddings. `train` reuses those caches
and trains only the multimodal components defined by the configuration.

### Hard negatives

```powershell
.venv\Scripts\shopee-hard-negatives mine --config configs\experiment\hard_negative_pair_head_pilot.yaml
.venv\Scripts\shopee-hard-negatives train --config configs\experiment\hard_negative_pair_head_pilot.yaml
.venv\Scripts\shopee-hard-negatives summarize --config configs\experiment\hard_negative_repeated_seed_summary.yaml
```

### Candidate retrieval

```powershell
.venv\Scripts\shopee-retrieval benchmark --config configs\experiment\candidate_retrieval_benchmark.yaml
```

Completed experiment outputs are immutable by default. A deliberate rerun should use a new
artifact root and report path or remove only the reviewed local ignored run. It should never
silently overwrite the evidence behind a published metric.

## 14. Artifact and provenance model

Version-controlled files contain source code, configurations, tests, lightweight figures, and
human-readable reports. Local-only artifacts contain competition data, split manifests, model
checkpoints, metrics JSON, embedding caches, hard-negative pair manifests, FAISS indexes, and
temporary diagnostics.

Reports link every material claim to a configuration and, where relevant, SHA-256 fingerprints of
the checkpoint and metrics that produced it. This allows a reviewer to distinguish recorded facts
from narrative interpretation without placing large or licensed files in Git.

## 15. Relationship to public competition solutions

Public high-performing solutions commonly combine pretrained image encoders, ArcFace-style metric
learning, multilingual transformer text embeddings, TF-IDF, threshold ensembles, and graph or
neighborhood post-processing. This project intentionally takes a different route through Phase 8:
its core encoders are compact, repository-owned, and trained from random initialization so that
the complete learning and retrieval pipeline can be explained and tested.

External scores are useful context, not direct leaderboard comparisons. Repositories use different
splits, K values, preprocessing, pretrained weights, and F1 definitions. The fair conclusion at
this stage is qualitative: the current scratch system reaches a credible retrieval range and has
strong protocol controls, while pretrained ensembles are expected to provide a higher quality
ceiling. A controlled pretrained comparison on this repository's frozen protocol is reserved for
Phase 9.

See [the competition solution review](../reports/competition_solution_review.md) for the referenced
methods and source links.

## 16. Known limitations

- The strongest held-out classical fusion still slightly outperforms the learned scratch model.
- Phase 6 gains are consistent but small and have not been measured on test.
- Phase 7 latency is measured on the current local hardware and validation catalog size.
- Label noise can cause genuine matches to be treated as hard negatives despite conservative
  guards.
- Candidate recall does not guarantee pair precision or cluster quality.
- No final graph clustering, false-merge control, calibration, or review policy exists yet.
- Pretrained baselines and full three-seed final evaluation are intentionally deferred.

## 17. Next stage: Phase 8

Phase 8 will consume the Top-50 candidate sets and complete the identity-resolution layer:

1. apply the frozen pair head or calibrated fused score to candidate pairs;
2. choose thresholds using validation only;
3. create conservative reciprocal-neighbor graph edges;
4. build candidate clusters with union-find or connected components;
5. add consistency rules to limit transitive false merges;
6. report pair metrics separately from B-cubed and pairwise clustering metrics;
7. categorize false merges, false splits, and manual-review cases.

Only after this stage is stable should the project start the Phase 9 pretrained comparison.
