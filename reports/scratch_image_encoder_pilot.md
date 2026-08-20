# Scratch image encoder pilot benchmark

## Decision

The bounded Phase 3 pilot is complete on the frozen validation split. Increasing product diversity
per batch from `P=8, K=2` to `P=16, K=2` was the only major experimental change and improved every
reported retrieval metric. The `P=16, K=2` sampler is therefore selected for the full-training
configuration. The test split remains locked.

This is a pilot decision, not the final Phase 3 benchmark. The full 224-pixel run must start from a
clean Git commit and is intentionally pending repository-owner version control.

## Controlled experiment

Both pilots used the repository-owned residual CNN with random initialization, 128-pixel inputs,
supervised contrastive loss, 12 epochs, 75 batches per epoch, seed `2026`, and exact cosine
validation retrieval. Both selected epoch 11. The second run changed only the number of distinct
products per batch.

| Metric | Smoke (`P=4`) | Pilot (`P=8`) | Diversity pilot (`P=16`) |
|---|---:|---:|---:|
| Validation mAP@20 | 0.26269 | 0.30032 | **0.34738** |
| Recall@1 | 0.15198 | 0.16386 | **0.18678** |
| Recall@5 | 0.24891 | 0.29203 | **0.33072** |
| Recall@10 | 0.27712 | 0.33655 | **0.38697** |
| Recall@20 | 0.29871 | 0.37848 | **0.43891** |
| Hit@20 | — | — | 0.59854 |
| Exact-positive-pHash mAP@20 | 0.67695 | 0.68700 | **0.72553** |
| No-exact-positive-pHash mAP@20 | 0.12169 | 0.16871 | **0.21866** |

The `P=16` pilot improved mAP@20 by 15.7% relative to `P=8`. More importantly, the
no-exact-positive-pHash stratum improved by 29.6% relative. This indicates that training learned
more than a lookup rule for visually identical images. The positive and negative cosine means
were `0.837` and `0.601`, respectively, compared with `0.874` and `0.692` for `P=8`; the lower
negative similarity is evidence of better class separation.

## Fair classical comparison

| Image-only retriever | Validation mAP@20 | Validation Recall@20 |
|---|---:|---:|
| Supplied pHash | 0.2895 | 0.3174 |
| ORB | **0.6638** | **0.8284** |
| Scratch CNN, bounded `P=16` pilot | 0.3474 | 0.4389 |

The pilot already exceeds the supplied pHash baseline, but remains substantially below ORB. It is
not fair to treat this bounded 128-pixel experiment as the final neural result: it used only 900
optimization batches and its best checkpoint occurred at the final epoch, so convergence had not
plateaued. Conversely, the gap to ORB is too large to claim that more training will necessarily
close it. The full run is needed to measure that hypothesis.

## Stratified diagnostics

The selected pilot achieved mAP@20 of `0.2931`, `0.3754`, `0.4362`, and `0.3554` for group-size
bands 2, 3–5, 6–9, and 10+, respectively. Two-listing groups remain hardest because each query has
only one relevant partner and a single miss has maximum impact. Performance is strongest for
groups of 6–9, where more positive examples expose more stable product-level visual cues.

Manual review of the first ten top-1 false matches found two recurring mechanisms:

- **Global-layout shortcuts:** unrelated items were matched through similar white backgrounds,
  centered dark objects, rectangular packaging, advertisement layouts, or dense product collages.
- **Coarse-category hard negatives:** bags matched other bags and one instant-noodle product
  matched a different noodle product, despite not being the same exact purchasable item.

The first mechanism dominated this small review. The encoder is learning useful similarity but
still relies too heavily on background, color blocks, silhouette, and merchandising composition.
The full 224-pixel configuration targets this measured limitation by preserving finer packaging
and product-detail evidence; it does not change the architecture or introduce title information.
The local contact sheet and raw review records remain ignored and are not distributed.

## Efficiency and full-run gate

The selected 128-pixel pilot extracted validation embeddings at `314.96 listings/s`, used about
`447 MiB` peak CUDA memory, and produced a checkpoint of about `36.9 MB`. A one-epoch 224-pixel
runtime probe with `P=16, K=2` used about `1.01 GiB` peak CUDA memory. It estimates approximately
`8.62 minutes/epoch`, or `5.74 hours` for the configured 40 epochs on the NVIDIA RTX 4060.

The CUDA environment is verified with PyTorch `2.7.0+cu126`. Deterministic CUDA training now sets
`CUBLAS_WORKSPACE_CONFIG=:4096:8`, and all 45 tests pass.

The remaining gate is procedural and reproducibility-critical: the full run must record a clean
implementation commit. No commit or push is performed automatically; once the repository owner
reviews and commits the intended Phase 3 changes, the frozen run can start from
`configs/experiment/image_embedding_training.yaml`.
