# Multimodal fusion full-training summary

## Outcome

The selected Phase 5 residual fusion and pair-head system completed its full validation-only
training schedule. It selected epoch 1 by pair-head reranked mAP@20 and stopped after 7 of 30
configured epochs when six later epochs failed to improve the selection metric. Held-out test was
not accessed.

## Why Recall@20 is credible

The simple-fusion validation Recall@20 of `0.94005` is calculated over all 3,430 validation
listings, using the complete validation split as the candidate pool and excluding the query itself.
The split loader rejects any `label_group` that crosses train, validation, and test. Frozen image
and text checkpoints were trained on train only; validation labels were used to select the fusion
weight and checkpoint, not to construct embeddings.

The result is also consistent with the independently implemented Phase 2 classical fusion, which
reached validation Recall@20 `0.9411`. It is therefore a plausible validation result rather than
evidence of a fabricated score. It must not be presented as a test result because the image weight
`0.40` was selected on validation.

## Loss-weight ablation

All learned rows use the same seed, frozen caches, residual architecture, sampler, optimizer,
learning rate, pair-negative ratio, and validation protocol. Only the pair-BCE weight changes.

| System | Pair-BCE weight | mAP@20 | Recall@20 | Pair F1 |
|---|---:|---:|---:|---:|
| Simple score fusion | — | 0.87358 | **0.94005** | 0.70437 |
| Residual pair head | 0.50 | 0.87565 | 0.93666 | 0.70274 |
| Residual pair head | 0.25 | 0.87868 | 0.93813 | 0.71081 |
| Residual pair head | **0.10** | **0.87903** | 0.93780 | **0.71285** |

Weight `0.10` is selected because it leads both validation mAP@20 and pair F1. Relative to simple
fusion, it improves mAP@20 by `0.00545` and pair F1 by `0.00848`, while Recall@20 decreases by
`0.00225`. This is a measured ranking/classification gain with a small candidate-recall trade-off,
not an across-the-board improvement.

## Full-training result

| Method | Validation mAP@20 | Validation Recall@20 | Pair F1 |
|---|---:|---:|---:|
| Frozen image encoder | 0.53907 | 0.64667 | — |
| Frozen text encoder | 0.75693 | 0.87414 | — |
| Simple score fusion | 0.87358 | **0.94005** | 0.70437 |
| Selected residual joint embedding | 0.87023 | 0.93780 | 0.70185 |
| Selected residual pair-head rerank | **0.87903** | 0.93780 | **0.71285** |

- Selected epoch: `1`.
- Pair threshold selected on validation: `0.244171`.
- Pair precision / recall: `0.74557 / 0.68289`.
- Trainable parameters: `1,051,137`.
- Checkpoint size: `12,629,805` bytes.
- Fusion-training wall time, excluding reusable cache preparation: `23.08` seconds.
- Completed/configured epochs: `7 / 30`, stopped early.

Training loss continued to fall after epoch 1 while validation pair-head mAP fell from `0.87903`
to `0.84779` by epoch 7. Retaining the earliest checkpoint is therefore necessary and confirms
that lower optimization loss does not automatically mean better retrieval quality.

## Frozen artifact identifiers

- Full checkpoint SHA-256: `95289d84fbb85f99764f42b05ded92ec2c535b2b421b3fa1422cfb987b2800f4`.
- Full training config SHA-256: `2a0d9912668fa0758bffc13e16a34791c3708cee02063ae4b6d25bc4956bab53`.
- Refreshed training metrics SHA-256: `980d13cbc7723dd2462b88ccfdbd4b61434928b794600f29adb81b0dcb25884e`.

These hashes identify the current validation-selected artifacts. A dedicated frozen-test config
has not yet been created, and test remains disabled.

## Remaining Phase 5 gates

Full training is complete, but Phase 5 remains open until modality-disagreement failures are
categorized, the selected artifacts/protocol are frozen in a one-time test configuration, and the
value of repeated seeds is assessed.

