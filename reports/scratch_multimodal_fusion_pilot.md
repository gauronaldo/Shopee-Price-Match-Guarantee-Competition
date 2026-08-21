# Scratch multimodal fusion implementation

## Experiment contract

The Phase 5 `pilot` experiment reuses the frozen scratch image and text encoders from Phases 3-4.
Their deterministic embeddings are cached once. Only the randomly initialized fusion projection
and symmetric pair head are optimized; held-out test remains disabled.

## Validation comparison

| Method | mAP@20 | Recall@20 |
|---|---:|---:|
| Frozen image only | 0.53907 | 0.64667 |
| Frozen text only | 0.75693 | 0.87414 |
| Simple score fusion | 0.87358 | 0.94005 |
| Learned fusion | 0.81244 | 0.90817 |
| Pair-head rerank | 0.80868 | 0.90817 |

- Selected simple-fusion image weight: `0.40`
- Selected learned checkpoint: epoch `12` by validation `map@20`
- Trainable parameters: `788,737`
- Test status: `disabled_until_checkpoint_and_protocol_are_frozen`

## Training history

| Epoch | Total loss | Contrastive | Pair BCE | Validation mAP@20 |
|---:|---:|---:|---:|---:|
| 1 | 0.29646 | 0.06940 | 0.45413 | 0.79242 |
| 2 | 0.17173 | 0.03610 | 0.27126 | 0.79771 |
| 3 | 0.07390 | 0.02411 | 0.09958 | 0.80310 |
| 4 | 0.04687 | 0.02298 | 0.04778 | 0.80280 |
| 5 | 0.03450 | 0.01995 | 0.02911 | 0.80678 |
| 6 | 0.02847 | 0.01758 | 0.02178 | 0.80632 |
| 7 | 0.02171 | 0.01326 | 0.01690 | 0.80878 |
| 8 | 0.02088 | 0.01396 | 0.01384 | 0.80923 |
| 9 | 0.02121 | 0.01460 | 0.01321 | 0.81066 |
| 10 | 0.02052 | 0.01411 | 0.01283 | 0.81164 |
| 11 | 0.02105 | 0.01457 | 0.01295 | 0.81179 |
| 12 | 0.01836 | 0.01253 | 0.01167 | 0.81244 |

## Interpretation

This `pilot` run is an engineering gate, not the final Phase 5 benchmark. It verifies
frozen-source reproducibility, loss/gradient flow, checkpoint selection, modality ablations,
pair-head behavior, and validation-only evaluation before a pilot or full fusion run is approved.
