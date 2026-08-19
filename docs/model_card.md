# Model card

## Candidate: scratch residual image encoder

Status: **implemented and smoke-tested; not yet a frozen benchmark model**.

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

The real-data smoke run is an engineering gate, not a quality claim. It used four optimization
steps at 64 pixels and reached validation mAP@20 `0.26269`; it must not be compared as if it were
the full Phase 3 model. The bounded pilot and categorized image-only failure review remain open.
