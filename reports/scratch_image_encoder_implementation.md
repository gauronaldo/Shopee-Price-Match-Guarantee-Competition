# Scratch image encoder implementation review

## Outcome

The Phase 3 image-only system is implemented end to end, but Phase 3 is not closed. The repository
can now train a compact residual encoder from random initialization, select checkpoints on the
frozen validation split, evaluate exact cosine retrieval, and emit reproducibility and diagnostic
artifacts without reading title features or evaluating the test split.

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

All 44 unit and integration tests passed after implementation. New coverage includes tensor
shapes, unit embedding norms, finite gradients, seeded random initialization, serialization
parity, supervised contrastive toy cases, deterministic batch composition, decode/preprocessing,
tiny-batch overfit, exact retrieval, diagnostics, config rejection rules, and an end-to-end
synthetic training run.

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

## Compute gate

The machine exposes an NVIDIA RTX 4060 with 8 GB VRAM, but the verified environment currently has
PyTorch `2.7.0+cpu`. Two attempts to download the official CUDA 12.6 wheel timed out because the
wheel is approximately 2.77 GB. The CPU environment remained intact and all checks still pass.

The bounded pilot configuration is ready at
`configs/experiment/image_embedding_pilot.yaml`, but it was intentionally not run on CPU. The
smoke profile shows that validation extraction alone costs roughly 20–25 seconds at 64 pixels;
the 128-pixel, 12-epoch pilot would repeatedly pay a substantially larger extraction cost. The
full 224-pixel, 40-epoch run must not begin until the pilot demonstrates stable non-collapsed
embeddings and the CUDA environment is verified.

## Remaining Phase 3 gates

1. Install and verify a CUDA-enabled PyTorch wheel.
2. Run the bounded pilot and inspect loss, positive/negative cosine distributions, stratified
   retrieval metrics, and nearest-neighbor failures.
3. Change only one major factor if the pilot fails; do not jump directly to a larger backbone.
4. Freeze one full-training configuration from validation evidence.
5. Run full training from a clean Git commit and reproduce the selected checkpoint metric.
6. Manually categorize the generated review manifest using the repository error taxonomy.
7. Only then unlock the one-time test evaluation and decide whether Phase 3 beats ORB or documents
   a measured quality/latency trade-off.
