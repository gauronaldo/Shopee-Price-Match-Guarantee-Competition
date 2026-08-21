# Scratch multimodal fusion implementation

## Experiment contract

The Phase 5 `training` experiment reuses the frozen scratch image and text encoders from Phases 3-4.
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
| 2 | 0.02692 | 0.00752 | 0.19400 | 0.85896 |
| 3 | 0.01278 | 0.00659 | 0.06190 | 0.85528 |
| 4 | 0.00955 | 0.00674 | 0.02808 | 0.85187 |
| 5 | 0.00812 | 0.00657 | 0.01551 | 0.85222 |
| 6 | 0.00899 | 0.00778 | 0.01213 | 0.85025 |
| 7 | 0.00645 | 0.00559 | 0.00861 | 0.84779 |

## Interpretation

This full validation-only run completed `7` of
`30` configured epochs and stopped early after the selected metric failed to improve.
The best checkpoint is retained independently of the lower later training loss. Held-out test
remains disabled until the checkpoint, threshold, and protocol are frozen.
