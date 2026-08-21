# Scratch multimodal fusion implementation

## Experiment contract

The Phase 5 `seed_2028` experiment reuses the frozen scratch image and text encoders from Phases 3-4.
Their deterministic embeddings are cached once. Only the randomly initialized fusion projection
and symmetric pair head are optimized; held-out test remains disabled.

## Validation comparison

| Method | mAP@20 | Recall@20 |
|---|---:|---:|
| Frozen image only | 0.53907 | 0.64667 |
| Frozen text only | 0.75693 | 0.87414 |
| Simple score fusion | 0.87358 | 0.94005 |
| Learned fusion | 0.86948 | 0.93894 |
| Pair-head rerank | 0.87965 | 0.93894 |

- Selected simple-fusion image weight: `0.40`
- Selected checkpoint: `epoch 1` by validation `pair_head_rerank.map@20`
- Trainable parameters: `1,051,137`
- Test status: `disabled_until_checkpoint_and_protocol_are_frozen`

## Training history

| Epoch | Total loss | Contrastive | Pair BCE | Validation mAP@20 |
|---:|---:|---:|---:|---:|
| 1 | 0.04705 | 0.00871 | 0.38333 | 0.87965 |
| 2 | 0.02874 | 0.00863 | 0.20109 | 0.86647 |
| 3 | 0.01730 | 0.01060 | 0.06700 | 0.85699 |
| 4 | 0.01062 | 0.00781 | 0.02809 | 0.85433 |
| 5 | 0.00869 | 0.00707 | 0.01622 | 0.85471 |
| 6 | 0.00877 | 0.00755 | 0.01222 | 0.85813 |
| 7 | 0.00873 | 0.00787 | 0.00862 | 0.85769 |

## Interpretation

This `seed_2028` run is an engineering gate, not the final Phase 5
benchmark. It verifies frozen-source reproducibility, loss/gradient flow, checkpoint selection,
modality ablations, pair-head behavior, and validation-only evaluation before a full fusion run is
approved.
