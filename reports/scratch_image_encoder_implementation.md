# Scratch image encoder implementation review

## Outcome

The Phase 3 image-only system is implemented and evaluated end to end. The repository can train a
compact residual encoder from random initialization, select checkpoints on the frozen validation
split, evaluate exact cosine retrieval, and emit reproducibility and diagnostic artifacts without
using title features.

## Implemented contract

- OpenCV image decode with explicit BGR-to-RGB conversion.
- Deterministic aspect-preserving resize/pad and fixed non-pretrained normalization.
- Conservative train augmentation: horizontal flip, rotation up to five degrees, small
  translation, mild contrast/brightness changes, and low-amplitude noise.
- Deterministic product-aware `P × K` batches with within-product replacement when a group is
  smaller than `K`.
- Repository-owned residual CNN, global average pooling, projection head, and L2-normalized
  embedding.
- Explicit random initialization; unsupported sources and pretrained checkpoints are rejected.
- Supervised contrastive loss, AdamW, cosine learning-rate schedule, gradient clipping, early
  stopping, atomic best/latest checkpoints, and resume metadata.
- Validation-only checkpoint selection using mAP@20; test evaluation is rejected by config.
- Exact cosine retrieval with Recall/Precision/F1/Hit-rate at K, mAP@20, pair-threshold selection,
  p50/p95 ranking latency, group-size strata, exact-positive-pHash strata, bounded similarity
  distributions, and a local nearest-neighbor review manifest.

## Verification

All 51 unit and integration tests pass after implementation and the deterministic CUDA update. New
coverage includes tensor shapes, unit embedding norms, finite gradients, seeded random
initialization, serialization parity, supervised contrastive toy cases, deterministic batch
composition, decode/preprocessing, tiny-batch overfit, exact retrieval, diagnostics, config
rejection rules, and an end-to-end synthetic training run.

The real split smoke run used seed `2026`, image size `64`, two epochs, two `P=4, K=2` batches per
epoch, and CPU PyTorch. It selected epoch 0:

| Metric | Smoke result |
|---|---:|
| Validation mAP@20 | 0.26269 |
| Validation Recall@1 | 0.15198 |
| Validation Recall@5 | 0.24891 |
| Validation Recall@10 | 0.27712 |
| Validation Recall@20 | 0.29871 |
| Parameters | 3,060,000 |
| Checkpoint size | 36,842,223 bytes |
| Validation extraction throughput | 167.28 listings/s |
| Exact-ranking p50 / p95 latency | 0.491 / 0.680 ms/query |
| End-to-end smoke wall time | 66.53 s |

This score is not a Phase 3 benchmark. The model saw only four optimization batches; the run
exists to prove data loading, forward/backward, checkpoint selection, and full validation
retrieval work without leakage. It is therefore not a fair comparison against the full Phase 2
ORB validation mAP@20 of `0.6638`.

The diagnostic outputs are also behaving as intended. Positive-pair cosine similarity averaged
`0.878`, compared with `0.740` for a bounded sample of negative pairs. The gap shows the nearly
untrained encoder is not completely collapsed, but the high negative median (`0.800`) indicates
weak separation and strongly overlapping distributions. This is expected after four updates and
must improve in the pilot.

The smoke result is highly dependent on exact visual duplication: queries having an exact-pHash
positive reached mAP@20 `0.67695`, while queries without one reached only `0.12169`. Groups of two
were also the weakest group-size band at mAP@20 `0.21598`. These are useful sanity signals rather
than conclusions about the final architecture: the pilot must demonstrate improvement on the
no-exact-positive stratum instead of merely relearning near-duplicate image matching.

## Pilot and compute gate

The CUDA environment is now verified on the NVIDIA RTX 4060 using PyTorch `2.7.0+cu126`.
Deterministic CUDA operation sets `CUBLAS_WORKSPACE_CONFIG=:4096:8` before CUDA initialization.

Two controlled 128-pixel pilots completed successfully. The original `P=8, K=2` run reached
validation mAP@20 `0.30032`; changing only product diversity to `P=16, K=2` raised it to `0.34738`
and raised Recall@20 from `0.37848` to `0.43891`. Both selected their final epoch, so the bounded
runs had not plateaued. The detailed comparison and error analysis are in
[`scratch_image_encoder_pilot.md`](scratch_image_encoder_pilot.md).

A 224-pixel runtime probe used about 1.01 GiB peak CUDA memory and estimated 8.62 minutes per full
epoch, or 5.74 hours for 40 epochs. The completed full run selected epoch 39 with validation
mAP@20 `0.53907` and Recall@20 `0.64667`.

## Final outcome

The frozen checkpoint reached test mAP@20 `0.55674`, Recall@20 `0.65941`, and pair F1 `0.48973` at
the validation-selected threshold. It clearly exceeds supplied pHash and remains below the
candidate-assisted ORB pipeline. The final comparison and categorized failure analysis document
that trade-off, so Phase 3 is closed.
