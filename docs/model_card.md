# Model card

## Candidate: scratch residual image encoder

Status: **implemented and pilot-evaluated; full benchmark training pending**.

- Intended phase: image-only exact-product retrieval in Phase 3.
- Architecture: repository-owned residual CNN, global average pooling, two-layer projection head,
  and L2-normalized 256-dimensional embedding.
- Initialization: explicit random Kaiming initialization; pretrained checkpoints are rejected by
  configuration validation.
- Objective: supervised contrastive loss over deterministic product-aware `P × K` batches.
- Inputs: RGB listing images with aspect-preserving resize/pad and fixed non-pretrained
  half-range normalization.
- Selection: validation mAP@20 on the frozen group-disjoint split.
- Test policy: disabled until the pilot, checkpoint, and retrieval protocol are frozen.

The real-data smoke run was an engineering gate, not a quality claim. Two controlled 128-pixel
pilots then compared product diversity per batch. `P=16, K=2` reached validation mAP@20 `0.34738`
and Recall@20 `0.43891`, outperforming `P=8, K=2` at `0.30032 / 0.37848`. It also raised mAP@20
for queries without an exact-pHash positive from `0.16871` to `0.21866`. This configuration is
selected for full training, but it is not yet the frozen benchmark checkpoint.

The pilot exceeds the supplied pHash baseline (`0.2895` mAP@20) but remains well below ORB
(`0.6638`). Full 224-pixel training and the final categorized review remain open. Test evaluation
is disabled until the full checkpoint and retrieval protocol are frozen.
