# Scratch multimodal fusion implementation

## Experiment contract

The Phase 5 `contrastive_only_ablation` experiment reuses the frozen scratch image and text encoders from Phases 3-4.
Their deterministic embeddings are cached once. Only the randomly initialized fusion projection
and symmetric pair head are optimized; held-out test remains disabled.

## Validation comparison

| Method | mAP@20 | Recall@20 |
|---|---:|---:|
| Frozen image only | 0.53907 | 0.64667 |
| Frozen text only | 0.75693 | 0.87414 |
| Simple score fusion | 0.87358 | 0.94005 |
| Learned fusion | 0.87358 | 0.94005 |
| Pair-head rerank | 0.40388 | 0.94005 |

- Selected simple-fusion image weight: `0.40`
- Selected checkpoint: `initialization` by validation `learned_fusion.map@20`
- Trainable parameters: `1,051,137`
- Test status: `disabled_until_checkpoint_and_protocol_are_frozen`

## Training history

| Epoch | Total loss | Contrastive | Pair BCE | Validation mAP@20 |
|---:|---:|---:|---:|---:|
| 1 | 0.00819 | 0.00819 | 0.68625 | 0.87114 |
| 2 | 0.00757 | 0.00757 | 0.68664 | 0.86962 |
| 3 | 0.00666 | 0.00666 | 0.68664 | 0.87044 |
| 4 | 0.00678 | 0.00678 | 0.68691 | 0.86941 |
| 5 | 0.00658 | 0.00658 | 0.68712 | 0.87036 |
| 6 | 0.00786 | 0.00786 | 0.68663 | 0.86851 |

## Interpretation

This `contrastive_only_ablation` run is an engineering gate, not the final Phase 5
benchmark. It verifies frozen-source reproducibility, loss/gradient flow, checkpoint selection,
modality ablations, pair-head behavior, and validation-only evaluation before a full fusion run is
approved.
