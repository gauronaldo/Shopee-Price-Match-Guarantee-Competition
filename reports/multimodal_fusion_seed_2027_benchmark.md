# Scratch multimodal fusion implementation

## Experiment contract

The Phase 5 `seed_2027` experiment reuses the frozen scratch image and text encoders from Phases 3-4.
Their deterministic embeddings are cached once. Only the randomly initialized fusion projection
and symmetric pair head are optimized; held-out test remains disabled.

## Validation comparison

| Method | mAP@20 | Recall@20 |
|---|---:|---:|
| Frozen image only | 0.53907 | 0.64667 |
| Frozen text only | 0.75693 | 0.87414 |
| Simple score fusion | 0.87358 | 0.94005 |
| Learned fusion | 0.87166 | 0.93841 |
| Pair-head rerank | 0.88157 | 0.93841 |

- Selected simple-fusion image weight: `0.40`
- Selected checkpoint: `epoch 1` by validation `pair_head_rerank.map@20`
- Trainable parameters: `1,051,137`
- Test status: `disabled_until_checkpoint_and_protocol_are_frozen`

## Training history

| Epoch | Total loss | Contrastive | Pair BCE | Validation mAP@20 |
|---:|---:|---:|---:|---:|
| 1 | 0.04679 | 0.00796 | 0.38834 | 0.88157 |
| 2 | 0.02891 | 0.00897 | 0.19942 | 0.86648 |
| 3 | 0.01514 | 0.00869 | 0.06450 | 0.85717 |
| 4 | 0.01238 | 0.00956 | 0.02815 | 0.86267 |
| 5 | 0.00960 | 0.00791 | 0.01698 | 0.85682 |
| 6 | 0.00807 | 0.00684 | 0.01224 | 0.85798 |
| 7 | 0.01121 | 0.01019 | 0.01022 | 0.85434 |

## Interpretation

This `seed_2027` run is an engineering gate, not the final Phase 5
benchmark. It verifies frozen-source reproducibility, loss/gradient flow, checkpoint selection,
modality ablations, pair-head behavior, and validation-only evaluation before a full fusion run is
approved.
