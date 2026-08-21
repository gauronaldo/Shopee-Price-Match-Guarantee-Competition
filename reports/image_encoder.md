# Custom image encoder

## Outcome

Phase 3 implements and evaluates an image-only residual encoder trained from random
initialization. The frozen model reaches validation/test mAP@20 of `0.53907 / 0.55674` and
Recall@20 of `0.64667 / 0.65941`. It clearly improves on supplied pHash, but remains below the
candidate-assisted ORB pipeline. No title, pHash, ORB score, or pretrained weight enters the model.

## System design

- OpenCV decode with explicit BGR-to-RGB conversion and aspect-preserving resize/pad.
- Conservative crop, flip, rotation, translation, brightness/contrast, and noise augmentation.
- Repository-owned residual CNN, global average pooling, projection head, and normalized
  512-dimensional product embedding.
- Product-aware `P x K` batches, supervised contrastive loss, AdamW, cosine scheduling, gradient
  clipping, atomic checkpoints, and validation-only checkpoint selection.
- Exact full-split cosine retrieval with self-exclusion and validation-selected pair threshold.

The model has `3,060,000` parameters. The full configuration uses 224-pixel inputs and
`P=16, K=2`; its measured peak allocated CUDA memory was approximately `1.01 GiB` on an NVIDIA
RTX 4060.

## Experiment progression

| Stage | Resolution / batches | Validation mAP@20 | Validation Recall@20 | Decision |
|---|---|---:|---:|---|
| Smoke | 64 px, four updates | 0.26269 | 0.29871 | Pipeline and checkpoint gate only |
| Pilot | 128 px, `P=8, K=2` | 0.30032 | 0.37848 | Insufficient product diversity |
| Diversity pilot | 128 px, `P=16, K=2` | 0.34738 | 0.43891 | Promoted to full training |
| Full validation | 224 px, 40 epochs | **0.53907** | **0.64667** | Epoch 39 selected |
| Frozen test | Locked checkpoint/protocol | **0.55674** | **0.65941** | One-time evaluation |

Increasing product diversity from `P=8` to `P=16` improves pilot mAP by 15.7% relative and raises
the no-exact-positive-pHash stratum from `0.16871` to `0.21866`. This is evidence that the model
learns beyond a lookup rule for visually identical images.

## Full-training trend

| Epoch | Train loss | Validation mAP@20 |
|---:|---:|---:|
| 0 | 2.30263 | 0.31742 |
| 5 | 1.37675 | 0.42069 |
| 10 | 1.01459 | 0.47075 |
| 15 | 0.77609 | 0.49379 |
| 20 | 0.58305 | 0.51296 |
| 25 | 0.45374 | 0.52802 |
| 30 | 0.35540 | 0.52822 |
| 35 | 0.32759 | 0.53517 |
| 39 | 0.30881 | **0.53907** |

The best checkpoint occurs at the final epoch, but the gain over the final five epochs is only
`0.00314`; simply adding more epochs is not the most evidence-based next improvement.

## Retrieval comparison

| Method | Validation mAP@20 | Validation Recall@20 | Test mAP@20 | Test Recall@20 |
|---|---:|---:|---:|---:|
| Supplied pHash | 0.2895 | 0.3174 | 0.3073 | 0.3345 |
| Custom residual CNN | **0.5391** | **0.6467** | **0.5567** | **0.6594** |
| ORB candidate pipeline | 0.6638 | 0.8284 | 0.6577 | 0.8151 |

The ORB result is not a pure image-only comparison because its candidate union includes TF-IDF.
The custom CNN performs exact image-embedding search over the complete evaluation split.

At the frozen validation threshold `0.805664`, test pair precision/recall/F1 are
`0.83231 / 0.34693 / 0.48973`.

## Failure analysis

The deterministic validation review contains 20 Top-1 false matches and 20 complete Top-20
retrieval misses. Percentages describe this bounded review set, not the whole split.

| Top-1 false-match category | Count | Share |
|---|---:|---:|
| Shared layout, background, color block, or coarse silhouette | 14 | 70% |
| Coarse-category hard negative | 4 | 20% |
| Probable label fragmentation | 2 | 10% |

| Retrieval-miss category | Count | Share |
|---|---:|---:|
| Shared layout/background/silhouette displaces the positive | 15 | 75% |
| Category or brand-family hard negative | 5 | 25% |

Validation mAP@20 is `0.83327` when an exact-pHash positive exists and `0.43893` when it does not.
The main image-only weakness is therefore visually different listings of the same product, plus
global-layout shortcuts that confuse commercially distinct variants.

## Efficiency and frozen evidence

| Measurement | Validation | Test |
|---|---:|---:|
| Embedding throughput | 270.50 listings/s | 159.93 listings/s |
| Exact ranking p50 | 0.442 ms/query | 0.450 ms/query |
| Exact ranking p95 | 0.656 ms/query | 0.717 ms/query |

- Checkpoint SHA-256: `6ea26b493d643b148cbcc48006231637b266491a0a026d7fdbd22284f7100e07`
- Canonical training-config SHA-256: `93286c4c66cd68a0d72c2f5894ac6a8347208892267c269ba59bc3f46fee3bd1`
- Training-metrics SHA-256: `3b389c5dd6cb58548249931fa77b1aa6b5821d540f07f9d99294e8725cee2a6a`

The checkpoint, threshold, and exact Top-20 protocol were frozen before the single test
evaluation. Phase 3 is closed.
