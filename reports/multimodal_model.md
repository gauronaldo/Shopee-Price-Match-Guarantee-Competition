# Custom multimodal model

## Outcome

Phase 5 combines frozen custom image and text embeddings with a score-preserving residual fusion
module and symmetric pair head. The canonical frozen-test result is mAP@20 `0.86848`, Recall@20
`0.93235`, and pair F1 `0.68429`. It beats either custom modality and classical TF-IDF retrieval on
mAP, but remains below the strongest classical fused pipeline.

Only the fusion projection and pair head are trained; both encoders remain frozen and were
originally trained from random initialization. Cached embeddings exclude test during training.

## Architecture and training contract

- Inputs: normalized image embedding `v` and text embedding `t`.
- Residual fusion starts from simple score-fusion geometry and learns a correction instead of
  discarding strong unimodal structure.
- Symmetric pair features use elementwise products and absolute differences so score order does
  not depend on pair direction.
- Combined objective: supervised contrastive loss plus pair BCE; validation mAP@20 selects the
  checkpoint and validation selects the pair threshold.
- Trainable parameters: `1,051,137`; checkpoint size: `12,629,805` bytes.

## Pilot and loss selection

| Experiment | Pair-BCE weight | Validation mAP@20 | Recall@20 | Pair F1 | Decision |
|---|---:|---:|---:|---:|---|
| Projected fusion pilot | 0.50 | 0.80868 | 0.90817 | 0.65554 | Rejected: loses baseline geometry |
| Residual fusion pilot | 0.50 | 0.87565 | 0.93666 | 0.70274 | Viable, but pair loss too strong |
| Residual fusion | 0.25 | 0.87868 | 0.93813 | 0.71081 | Improved |
| Residual fusion | **0.10** | **0.87903** | 0.93780 | **0.71285** | Selected |

The selected pair weight improves validation mAP by `+0.00545` and pair F1 by `+0.00848` over
simple fusion, with a `-0.00225` Recall@20 trade-off. Training loss continues falling after epoch
1 while validation ranking degrades, so early stopping is essential.

## Required ablations

| Configuration | Validation mAP@20 | Recall@20 | Interpretation |
|---|---:|---:|---|
| Image only | 0.53907 | 0.64667 | Visual evidence alone is insufficient |
| Text only | 0.75693 | 0.87414 | Stronger custom modality |
| Simple score fusion | 0.87358 | **0.94005** | Strong non-learned reference |
| Contrastive only | 0.87358 | 0.94005 | Best checkpoint is initialization |
| Pair BCE only | 0.85092 | 0.93827 | Pair supervision alone damages ranking |
| Combined losses, pair head off | 0.87023 | 0.93780 | Joint embedding trails simple fusion |
| Combined losses, pair head on | **0.87903** | 0.93780 | Best learned ranking |

## Canonical training history

| Epoch | Total loss | Contrastive | Pair BCE | Validation mAP@20 |
|---:|---:|---:|---:|---:|
| 1 | 0.04629 | 0.00819 | 0.38103 | **0.87903** |
| 2 | 0.02692 | 0.00752 | 0.19400 | 0.85896 |
| 3 | 0.01278 | 0.00659 | 0.06190 | 0.85528 |
| 4 | 0.00955 | 0.00674 | 0.02808 | 0.85187 |
| 5 | 0.00812 | 0.00657 | 0.01551 | 0.85222 |
| 6 | 0.00899 | 0.00778 | 0.01213 | 0.85025 |
| 7 | 0.00645 | 0.00559 | 0.00861 | 0.84779 |

## Repeated-seed validation

The runs differ only by initialization and deterministic seed-controlled sampling. Seed 2026
remains canonical; the best observed seed is not selected retroactively.

| Seed | Best epoch | Pair-head mAP@20 | Recall@1 | Recall@5 | Recall@10 | Recall@20 |
|---:|---:|---:|---:|---:|---:|---:|
| 2026 | 1 | 0.87903 | 0.48957 | 0.80954 | 0.88735 | 0.93780 |
| 2027 | 1 | 0.88157 | 0.49236 | 0.80884 | 0.88552 | 0.93841 |
| 2028 | 1 | 0.87965 | 0.49087 | 0.80525 | 0.88627 | 0.93894 |
| Mean | - | **0.88008** | **0.49094** | **0.80787** | **0.88638** | **0.93838** |
| Sample standard deviation | - | 0.00132 | 0.00139 | 0.00230 | 0.00092 | 0.00057 |

The low mAP standard deviation shows the result is not explained by one lucky initialization.

## Frozen validation-to-test comparison

| Method | Validation mAP@20 | Test mAP@20 | Validation Recall@20 | Test Recall@20 |
|---|---:|---:|---:|---:|
| Image only | 0.53907 | 0.55675 | 0.64667 | 0.65941 |
| Text only | 0.75693 | 0.74841 | 0.87414 | 0.86978 |
| Simple custom fusion | 0.87358 | 0.86277 | 0.94005 | 0.93098 |
| Learned fusion | 0.87023 | 0.85946 | 0.93780 | 0.93235 |
| Pair-head rerank | **0.87903** | **0.86848** | 0.93780 | **0.93235** |

At the frozen threshold `0.244170547`, test pair precision/recall/F1 are
`0.74840 / 0.63029 / 0.68429`. Classical TF-IDF reaches test mAP `0.8564`; the strongest classical
fusion reaches mAP `0.8810`, Recall `0.9349`, and pair F1 `0.7220`.

## Failure analysis

| Validation diagnostic | Queries | Share |
|---|---:|---:|
| Pair-head Top-1 false match | 332 | 9.68% |
| No true duplicate in pair-head Top-20 | 68 | 1.98% |
| Pair head regresses a correct simple-fusion Top-1 | 32 | 0.93% |
| Pair head rescues an incorrect simple-fusion Top-1 | 54 | 1.57% |
| Image correct while text is wrong at Top-1 | 366 | 10.67% |
| Text correct while image is wrong at Top-1 | 1,022 | 29.80% |
| Image/text Top-1 disagreement | 2,590 | 75.51% |
| False Top-1 with digit/unit conflict | 220 | 6.41% |

The pair head improves average ranking but can regress already-correct simple-fusion queries.
Digit, quantity, size, volume, and model-number conflicts directly motivate hard-negative mining.

## Efficiency and frozen evidence

- Test image/text/fusion throughput: `174.86 / 14,345.76 / 77,691.86` listings/s.
- Test exact-ranking p50/p95: `0.366 / 0.614 ms/query`.
- Checkpoint SHA-256: `95289d84fbb85f99764f42b05ded92ec2c535b2b421b3fa1422cfb987b2800f4`
- Canonical training-config SHA-256: `279c96794c207fb2e62e4638cdae315dc4ffd4a2b85ecf039f41861e7412377c`
- Training-metrics SHA-256: `da4d681cf0fb656905748a85c88f1285016ae4aee690826fdcd809521de4a313`

The checkpoint, simple-fusion weight, threshold, and Top-20 protocol were frozen before the
single test evaluation. Phase 5 is closed.
