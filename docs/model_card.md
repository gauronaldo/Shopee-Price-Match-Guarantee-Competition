# Model Card: Custom Multimodal Entity Resolution System

## Model details

- System version: `final.system_evaluation.v1`.
- Task: exact-product candidate retrieval, pair scoring, and duplicate-entity clustering.
- Inputs: one product image and one noisy multilingual product title per listing.
- Outputs: Top-K candidates, pair probabilities and modality evidence, predicted entity IDs,
  confidence, and manual-review flags.
- Framework: PyTorch with repository-owned image, text, fusion, pair-head, sampling, loss, and
  evaluation code.
- Initialization: all custom neural weights use random initialization. External pretrained weights
  are used only by the separately labeled EfficientNet comparison.

## Architecture

The custom image path is a 3.06-million-parameter residual CNN that produces a normalized
256-dimensional embedding. The custom text path is a 455-thousand-parameter character TextCNN
with a train-only vocabulary and normalized 256-dimensional embedding. A residual multimodal
fusion module maps image/text evidence to a normalized 512-dimensional listing embedding. A
symmetric pair head scores `[z1*z2, |z1-z2|]` so pair order cannot change the prediction.

Exact Top-50 cosine retrieval creates candidates. The pair head scores each unique candidate pair.
Reciprocal Top-5 edges above probability `0.16` enter a conservative union-find graph only when
full cross-component support is present. Clusters over 64 members are blocked, and low-confidence
clusters receive manual-review flags.

See [`architecture.md`](architecture.md) for training and inference diagrams.

## Training and selection

- Dataset: user-provided Kaggle Shopee Price Match Guarantee release.
- Split: deterministic group-disjoint train/validation/test manifest; labels, exact image hashes,
  filenames, and exact pHashes do not cross splits.
- Image objective: supervised contrastive loss with product-aware batches.
- Text objective: supervised contrastive loss over identity-preserving character tokens.
- Fusion objective: supervised contrastive loss plus pair binary cross-entropy.
- Hard negatives: train-only high-scoring cross-label pairs, with guards for same pHash and exact
  normalized titles.
- Selection: model checkpoints, candidate K, pair threshold, and graph policy use validation only.
- Repeated seeds: the main multimodal and accepted hard-negative experiments use seeds 2026-2028;
  seed 2026 remains canonical because it was pre-declared.

## Final quality

| Metric | Validation | Held-out test |
|---|---:|---:|
| Retrieval mAP@20 | 0.87023 | 0.85946 |
| Retrieval Recall@20 | 0.93780 | 0.93235 |
| Retrieval mAP@50 | 0.87279 | 0.86001 |
| Retrieval Recall@50 | 0.97438 | 0.96882 |
| Cluster pairwise precision | 0.90165 | 0.89591 |
| Cluster pairwise recall | 0.33119 | 0.32723 |
| Cluster pairwise F1 | 0.48444 | 0.47937 |
| B-cubed precision | 0.95618 | 0.95279 |
| B-cubed recall | 0.73003 | 0.72331 |
| B-cubed F1 | 0.82794 | 0.82234 |
| False-merge pair rate | 0.09835 | 0.10409 |
| False-split group rate | 0.30818 | 0.33637 |

On test candidate pairs, the raw pair head reaches average precision/PR-AUC `0.78497`, Brier score
`0.04992`, and expected calibration error `0.08596`. At the frozen graph operating point,
accepted-edge precision/recall/F1 are `0.81962 / 0.38345 / 0.52247` when every true test pair is
included in the recall denominator.

## Efficiency

Measured on the local CUDA environment and 3,429 held-out listings:

| Stage | Result |
|---|---:|
| Image extraction | 188.46 listings/s |
| Text extraction | 15,959.47 listings/s |
| Joint fusion | 83,475.14 listings/s |
| Pair scoring | 40,063.06 pairs/s |
| Exact query p50 / p95 | 0.344 / 0.416 ms |
| End-to-end evaluation wall time | 23.88 s |

Timings describe this hardware and catalog size; they are not production-scale guarantees.

## Intended use

- offline catalog deduplication research;
- candidate generation for human-assisted product-entity review;
- controlled comparison of classical, custom, and pretrained representations;
- portfolio demonstration of leakage-safe multimodal retrieval and clustering engineering.

## Out-of-scope use

- autonomous deletion or consolidation of marketplace records;
- legal or commercial product-identity decisions without human review;
- unseen marketplaces, languages, or catalog distributions without recalibration;
- production-readiness claims based solely on this benchmark or future demo.

## Limitations and risks

- Large heterogeneous product groups remain fragmented: no held-out group with at least 10
  listings is recovered as one entity under the precision-first graph policy.
- Similar packaging, brands, colors, and layouts can still merge different variants.
- Quantity, size, flavor, and model-number evidence may be missing or contradictory.
- Competition labels contain plausible fragmentation and variant-policy ambiguity.
- The test split was not used for final policy selection, but earlier component reports already
  disclosed results on the same split; it is not globally unseen to the project owner.
- Pair probabilities are not perfectly calibrated (`ECE 0.08596`) and should not be interpreted as
  universal commercial confidence.

## Provenance

- Final evaluation config SHA-256:
  `2f7741c3ec5a5e7032731029c2842f2219aae2a0e6b81d59eb5875fcc5d78d44`.
- Entity-resolution config SHA-256:
  `84b68e8478a237553e27cf41296ec9f47a1a146185d5657402e1330608a4c794`.
- Entity-resolution metrics SHA-256:
  `1d8c65a14d9cb9a4927bd3d0f56f7a7e2f7eab5e85f1a493bb856aa60b34fe1f`.
- Phase 6 checkpoint SHA-256:
  `d763834919c9bea2378b112e870d15b82817023692940c20f112f98d49370c3e`.
- Split manifest SHA-256:
  `c9cef390b5fbde6c833fddb15a0a8df2c7fbecacd8d50fb83aadba6056bf8e09`.
- Final test source commit: `f87639b8942020cbd0ba04a2113f3edb15f0d3d3`, clean worktree.

Aggregate final evidence is in [`../reports/final_evaluation.md`](../reports/final_evaluation.md).
Raw data, checkpoints, embeddings, indexes, pair manifests, and row-level reviews remain local and
ignored.
