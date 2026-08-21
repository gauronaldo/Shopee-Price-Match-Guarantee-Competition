# Scratch multimodal fusion implementation

## Experiment contract

The Phase 5 `pilot_pair_loss_010` experiment reuses the frozen scratch image and text encoders from Phases 3-4.
Their deterministic embeddings are cached once. Only the randomly initialized fusion projection
and symmetric pair head are optimized; held-out test remains disabled.

## Validation comparison

| Method | mAP@20 | Recall@20 |
|---|---:|---:|
| Frozen image only | 0.53907 | 0.64667 |
| Frozen text only | 0.75693 | 0.87414 |
| Simple score fusion | 0.87358 | 0.94005 |
| Learned fusion | 0.87023 | 0.93780 |
| Pair-head rerank | 0.87903 | 0.93780 |

- Selected simple-fusion image weight: `0.40`
- Selected checkpoint: `epoch 1` by validation `pair_head_rerank.map@20`
- Trainable parameters: `1,051,137`
- Test status: `disabled_until_checkpoint_and_protocol_are_frozen`

## Training history

| Epoch | Total loss | Contrastive | Pair BCE | Validation mAP@20 |
|---:|---:|---:|---:|---:|
| 1 | 0.04629 | 0.00819 | 0.38103 | 0.87903 |
| 2 | 0.02704 | 0.00752 | 0.19528 | 0.85926 |
| 3 | 0.01303 | 0.00658 | 0.06447 | 0.85523 |
| 4 | 0.00974 | 0.00671 | 0.03032 | 0.85164 |
| 5 | 0.00824 | 0.00649 | 0.01752 | 0.85194 |

## Interpretation

This `pilot_pair_loss_010` run is an engineering gate, not the final Phase 5 benchmark. It verifies
frozen-source reproducibility, loss/gradient flow, checkpoint selection, modality ablations,
pair-head behavior, and validation-only evaluation before a pilot or full fusion run is approved.
