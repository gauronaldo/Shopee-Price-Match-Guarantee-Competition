# Model card

## Candidate: scratch residual image encoder

Status: **Phase 3 complete; validation- and test-evaluated**.

- Intended phase: image-only exact-product retrieval in Phase 3.
- Architecture: repository-owned residual CNN, global average pooling, two-layer projection head,
  and L2-normalized 256-dimensional embedding.
- Initialization: explicit random Kaiming initialization; pretrained checkpoints are rejected by
  configuration validation.
- Objective: supervised contrastive loss over deterministic product-aware `P × K` batches.
- Inputs: RGB listing images with aspect-preserving resize/pad and fixed non-pretrained
  half-range normalization.
- Selection: validation mAP@20 on the frozen group-disjoint split.
- Test policy: evaluated once after the checkpoint, retrieval protocol, and validation threshold
  were frozen.

The real-data smoke run was an engineering gate, not a quality claim. Two controlled 128-pixel
pilots then compared product diversity per batch. `P=16, K=2` reached validation mAP@20 `0.34738`
and Recall@20 `0.43891`, outperforming `P=8, K=2` at `0.30032 / 0.37848`. It also raised mAP@20
for queries without an exact-pHash positive from `0.16871` to `0.21866`. This configuration is
selected for full training, but it is not yet the frozen benchmark checkpoint.

The full 224-pixel checkpoint reached validation/test mAP@20 `0.53907 / 0.55674` and Recall@20
`0.64667 / 0.65941`. It exceeds pHash clearly and remains below the ORB pipeline, whose candidate
union includes title TF-IDF. The validation threshold `0.805664` produced test pair precision,
recall, and F1 of `0.83231 / 0.34693 / 0.48973`.

The checkpoint, training config, and training metrics are SHA-256 locked. Test evaluation was
performed once without test-time threshold or hyperparameter selection.
