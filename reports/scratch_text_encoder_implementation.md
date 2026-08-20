# Scratch text encoder implementation review

## Outcome

The Phase 4 scratch text pipeline is complete. It fits a character vocabulary from training titles
only, trains a random-initialized multi-kernel TextCNN with supervised contrastive loss, selects
checkpoints on validation mAP@20, and evaluates the locked checkpoint on held-out test once.

## Implemented contract

- Shared NFKC/case-fold normalization that preserves letters, digits, and unit tokens.
- Explicit PAD and UNK tokens with deterministic train-only vocabulary fitting.
- Fixed-length encoding with unknown-character and truncation diagnostics.
- Character embedding plus convolution kernels 3, 5, and 7.
- Global max pooling, projection head, and normalized 256-dimensional title embedding.
- Shared deterministic product-aware `P × K` sampler.
- Supervised contrastive loss, AdamW, cosine schedule, gradient clipping, early stopping, and atomic
  best/latest checkpoints.
- Exact cosine retrieval with validation-only threshold selection.
- Retrieval diagnostics by group size and normalized title length.
- Concise progress messages controlled by `--progress-updates-per-epoch`.

## Smoke result

The smoke run used the real train/validation manifest, seed `2026`, two epochs, and two
`P=8, K=2` batches per epoch. It ran on the NVIDIA RTX 4060 and completed in approximately seven
seconds.

| Metric | Smoke result |
|---|---:|
| Validation mAP@20 | 0.64049 |
| Recall@1 | 0.37372 |
| Recall@5 | 0.61106 |
| Recall@10 | 0.67851 |
| Recall@20 | 0.73099 |
| Validation pair F1 | 0.55890 |
| Parameters | 455,040 |
| Checkpoint size | 5,479,523 bytes |
| Validation extraction throughput | 9,553.73 listings/s |
| Train-only vocabulary size | 40 |
| Validation unknown-character rate | 0.000000 |
| Validation truncation rate at 128 characters | 0.004956 |

The result is an engineering signal, not the Phase 4 benchmark. The model received only four
optimization batches, and epoch 0 remained the selected checkpoint. The relatively high score is
plausible because even a lightly trained character CNN preserves strong lexical overlap between
duplicate titles; it must not be compared as a completed experiment with the fully fitted TF-IDF
validation mAP@20 `0.8635`.

The weakest normalized-title-length band is 101+ characters at mAP@20 `0.47240`, compared with
`0.67959` for lengths 31–60. This motivates inspecting long-title truncation and noisy appended
keywords during the bounded pilot rather than increasing model size immediately.

## Final gate decision

The implementation, leakage, gradient, serialization, deterministic batching, tiny-batch overfit,
real-data smoke, pilot, and full-training gates pass. The full run completed all 30 epochs and
selected epoch 26 with validation mAP@20 `0.75698`. After the checkpoint, configuration, metrics,
threshold, and protocol were locked by SHA-256, the single test evaluation reached mAP@20
`0.74841`, Recall@20 `0.86978`, and pair F1 `0.56187`.

Phase 4 is closed. Character TF-IDF remains stronger, reaching test mAP@20 `0.8564`; the measured
gap and its failure categories are documented in
[`text_retrieval_final_comparison.md`](text_retrieval_final_comparison.md).
