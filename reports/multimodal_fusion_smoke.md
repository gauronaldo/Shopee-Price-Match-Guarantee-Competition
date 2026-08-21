# Scratch multimodal fusion implementation

## Experiment contract

The Phase 5 smoke experiment reuses the frozen scratch image and text encoders from Phases 3-4.
Their deterministic embeddings are cached once. Only the randomly initialized fusion projection
and symmetric pair head are optimized; held-out test remains disabled.

## Validation comparison

| Method | mAP@20 | Recall@20 |
|---|---:|---:|
| Frozen image only | 0.53907 | 0.64667 |
| Frozen text only | 0.75693 | 0.87414 |
| Simple score fusion | 0.86462 | 0.93509 |
| Learned fusion | 0.75871 | 0.85874 |
| Pair-head rerank | 0.72811 | 0.85874 |

- Selected simple-fusion image weight: `0.25`
- Selected learned checkpoint: epoch `1` by validation `map@20`
- Trainable parameters: `788,737`
- Test status: `disabled_until_checkpoint_and_protocol_are_frozen`

## Training history

| Epoch | Total loss | Contrastive | Pair BCE | Validation mAP@20 |
|---:|---:|---:|---:|---:|
| 1 | 0.44871 | 0.12542 | 0.64657 | 0.75871 |
| 2 | 0.31423 | 0.03350 | 0.56147 | 0.75091 |

## Interpretation

This bounded run is an engineering gate, not the final Phase 5 benchmark. It verifies frozen-source
reproducibility, loss/gradient flow, checkpoint selection, modality ablations, pair-head behavior,
and validation-only evaluation before a pilot or full fusion run is approved.
