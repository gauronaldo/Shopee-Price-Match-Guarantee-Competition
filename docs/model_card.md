# Model Card: Custom Multimodal Entity Resolution System

## Model details

- System version: `hybrid.system_evaluation.v1`.
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

Dense Top-50 cosine retrieval, train-fitted character TF-IDF Top-50, and pHash Top-20 candidates
are combined with weighted reciprocal-rank fusion and truncated at Top-75. The pair head scores
each unique candidate pair. Reciprocal Top-5 edges above probability `0.14` enter a conservative
union-find graph only when full cross-component support is present. A second pass can attach a
singleton at probability `0.18` when at least two members of the same component support it within
rank 50. Clusters over 64 members are blocked, and low-confidence clusters receive manual-review
flags.

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
| Mean sample-wise F1 | 0.81457 | 0.79591 |
| Retrieval mAP@20 | 0.90131 | 0.89245 |
| Retrieval Recall@20 | 0.95929 | 0.95371 |
| Retrieval Recall@75 | 0.99209 | 0.98615 |
| Cluster pairwise precision | 0.89582 | 0.87850 |
| Cluster pairwise recall | 0.45573 | 0.40396 |
| Cluster pairwise F1 | 0.60413 | 0.55344 |
| B-cubed precision | 0.95148 | 0.94269 |
| B-cubed recall | 0.78119 | 0.76913 |
| B-cubed F1 | 0.85797 | 0.84711 |
| False-merge pair rate | 0.10418 | 0.12150 |
| False-split group rate | 0.26364 | 0.28350 |

On test candidate pairs, the raw pair head reaches average precision/PR-AUC `0.76610`, Brier score
`0.03351`, and expected calibration error `0.06757`. At the frozen graph operating point,
accepted-edge precision/recall/F1 are `0.83803 / 0.38623 / 0.52876` when every true test pair is
included in the recall denominator.

## Efficiency

Measured on the local CUDA environment and 3,429 held-out listings:

| Stage | Result |
|---|---:|
| Image extraction | 19.73 s |
| Text extraction | 0.30 s |
| Joint fusion | 0.04 s |
| Hybrid retrieval | 34.92 s |
| Pair scoring | 37,000 pairs/s |
| Dense query p50 / p95 | 0.323 / 0.453 ms |
| End-to-end evaluation wall time | 62.05 s |

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
- Pair probabilities are not perfectly calibrated (`ECE 0.06757`) and should not be interpreted as
  universal commercial confidence.

## Provenance

- Final evaluation config SHA-256:
  `9f3c88a8e154101882821815681e2ce94cc3f2a3b9e0f5681ce0412022618957`.
- Entity-resolution config SHA-256:
  `982d6118bfde3b444d672a0b5717dc4833097bf0e0fd0fd09a3e53158cad083e`.
- Entity-resolution metrics SHA-256:
  `8833c0f6994b98dd08f10a4aa482d46e9520c25612b6b3e4be9ed6cd11a1472e`.
- Frozen checkpoint SHA-256:
  `d763834919c9bea2378b112e870d15b82817023692940c20f112f98d49370c3e`.
- Split manifest SHA-256:
  `c9cef390b5fbde6c833fddb15a0a8df2c7fbecacd8d50fb83aadba6056bf8e09`.
- Final test source commit: `51d109d626d444ca9b8e4dfd423a9459c08f6346`, clean worktree.

Aggregate final evidence is in [`../reports/final_evaluation.md`](../reports/final_evaluation.md).
Raw data, checkpoints, embeddings, indexes, pair manifests, and row-level reviews remain local and
ignored.
