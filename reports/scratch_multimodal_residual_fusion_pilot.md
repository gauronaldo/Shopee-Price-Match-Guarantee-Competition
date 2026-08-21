# Scratch multimodal fusion implementation

## Experiment contract

The Phase 5 `pilot_residual` experiment reuses the frozen scratch image and text encoders from Phases 3-4.
Their deterministic embeddings are cached once. Only the randomly initialized fusion projection
and symmetric pair head are optimized; held-out test remains disabled.

## Validation comparison

| Method | mAP@20 | Recall@20 |
|---|---:|---:|
| Frozen image only | 0.53907 | 0.64667 |
| Frozen text only | 0.75693 | 0.87414 |
| Simple score fusion | 0.87358 | 0.94005 |
| Learned fusion | 0.86651 | 0.93666 |
| Pair-head rerank | 0.87565 | 0.93666 |

- Selected simple-fusion image weight: `0.40`
- Selected checkpoint: `epoch 1` by validation `pair_head_rerank.map@20`
- Trainable parameters: `1,051,137`
- Test status: `disabled_until_checkpoint_and_protocol_are_frozen`

## Training history

| Epoch | Total loss | Contrastive | Pair BCE | Validation mAP@20 |
|---:|---:|---:|---:|---:|
| 1 | 0.19768 | 0.00855 | 0.37826 | 0.87565 |
| 2 | 0.10265 | 0.00802 | 0.18926 | 0.85479 |
| 3 | 0.03782 | 0.00696 | 0.06172 | 0.85113 |
| 4 | 0.02164 | 0.00691 | 0.02944 | 0.84907 |
| 5 | 0.01521 | 0.00661 | 0.01719 | 0.84998 |

## Interpretation

This `pilot_residual` run is an engineering gate, not the final Phase 5 benchmark. It verifies
frozen-source reproducibility, loss/gradient flow, checkpoint selection, modality ablations,
pair-head behavior, and validation-only evaluation before a pilot or full fusion run is approved.
