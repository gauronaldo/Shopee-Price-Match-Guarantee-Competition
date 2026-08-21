# Scratch multimodal fusion implementation

## Experiment contract

The Phase 5 `pilot_pair_loss_025` experiment reuses the frozen scratch image and text encoders from Phases 3-4.
Their deterministic embeddings are cached once. Only the randomly initialized fusion projection
and symmetric pair head are optimized; held-out test remains disabled.

## Validation comparison

| Method | mAP@20 | Recall@20 |
|---|---:|---:|
| Frozen image only | 0.53907 | 0.64667 |
| Frozen text only | 0.75693 | 0.87414 |
| Simple score fusion | 0.87358 | 0.94005 |
| Learned fusion | 0.86936 | 0.93813 |
| Pair-head rerank | 0.87868 | 0.93813 |

- Selected simple-fusion image weight: `0.40`
- Selected checkpoint: `epoch 1` by validation `pair_head_rerank.map@20`
- Trainable parameters: `1,051,137`
- Test status: `disabled_until_checkpoint_and_protocol_are_frozen`

## Training history

| Epoch | Total loss | Contrastive | Pair BCE | Validation mAP@20 |
|---:|---:|---:|---:|---:|
| 1 | 0.10320 | 0.00825 | 0.37980 | 0.87868 |
| 2 | 0.05567 | 0.00756 | 0.19243 | 0.85956 |
| 3 | 0.02240 | 0.00664 | 0.06302 | 0.85457 |
| 4 | 0.01421 | 0.00676 | 0.02982 | 0.85143 |
| 5 | 0.01084 | 0.00652 | 0.01728 | 0.85166 |

## Interpretation

This `pilot_pair_loss_025` run is an engineering gate, not the final Phase 5 benchmark. It verifies
frozen-source reproducibility, loss/gradient flow, checkpoint selection, modality ablations,
pair-head behavior, and validation-only evaluation before a pilot or full fusion run is approved.
