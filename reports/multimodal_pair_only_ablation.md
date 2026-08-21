# Scratch multimodal fusion implementation

## Experiment contract

The Phase 5 `pair_only_ablation` experiment reuses the frozen scratch image and text encoders from Phases 3-4.
Their deterministic embeddings are cached once. Only the randomly initialized fusion projection
and symmetric pair head are optimized; held-out test remains disabled.

## Validation comparison

| Method | mAP@20 | Recall@20 |
|---|---:|---:|
| Frozen image only | 0.53907 | 0.64667 |
| Frozen text only | 0.75693 | 0.87414 |
| Simple score fusion | 0.87358 | 0.94005 |
| Learned fusion | 0.86590 | 0.93827 |
| Pair-head rerank | 0.85092 | 0.93827 |

- Selected simple-fusion image weight: `0.40`
- Selected checkpoint: `epoch 19` by validation `pair_head_rerank.map@20`
- Trainable parameters: `1,051,137`
- Test status: `disabled_until_checkpoint_and_protocol_are_frozen`

## Training history

| Epoch | Total loss | Contrastive | Pair BCE | Validation mAP@20 |
|---:|---:|---:|---:|---:|
| 1 | 0.03693 | 0.03562 | 0.36929 | 0.78627 |
| 2 | 0.01867 | 0.05568 | 0.18666 | 0.82299 |
| 3 | 0.00645 | 0.01859 | 0.06448 | 0.83289 |
| 4 | 0.00302 | 0.01178 | 0.03018 | 0.83825 |
| 5 | 0.00167 | 0.01006 | 0.01671 | 0.83860 |
| 6 | 0.00128 | 0.00988 | 0.01279 | 0.84035 |
| 7 | 0.00089 | 0.00732 | 0.00893 | 0.84395 |
| 8 | 0.00073 | 0.00748 | 0.00730 | 0.84345 |
| 9 | 0.00059 | 0.00811 | 0.00589 | 0.84586 |
| 10 | 0.00075 | 0.00943 | 0.00750 | 0.84814 |
| 11 | 0.00066 | 0.01083 | 0.00664 | 0.84963 |
| 12 | 0.00049 | 0.00747 | 0.00486 | 0.84628 |
| 13 | 0.00036 | 0.00762 | 0.00355 | 0.84656 |
| 14 | 0.00038 | 0.00676 | 0.00382 | 0.84657 |
| 15 | 0.00042 | 0.00819 | 0.00425 | 0.84903 |
| 16 | 0.00044 | 0.00902 | 0.00441 | 0.85003 |
| 17 | 0.00036 | 0.00699 | 0.00364 | 0.84707 |
| 18 | 0.00031 | 0.00713 | 0.00311 | 0.84610 |
| 19 | 0.00042 | 0.00648 | 0.00416 | 0.85092 |
| 20 | 0.00034 | 0.00661 | 0.00337 | 0.84480 |
| 21 | 0.00034 | 0.00701 | 0.00336 | 0.84707 |
| 22 | 0.00029 | 0.00551 | 0.00285 | 0.84906 |
| 23 | 0.00026 | 0.00562 | 0.00262 | 0.84726 |
| 24 | 0.00030 | 0.00534 | 0.00303 | 0.84817 |
| 25 | 0.00035 | 0.00609 | 0.00347 | 0.84748 |

## Interpretation

This `pair_only_ablation` run is an engineering gate, not the final Phase 5
benchmark. It verifies frozen-source reproducibility, loss/gradient flow, checkpoint selection,
modality ablations, pair-head behavior, and validation-only evaluation before a full fusion run is
approved.
