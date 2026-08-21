# Multimodal fusion frozen test evaluation

## Locked protocol

The canonical seed-2026 checkpoint, training config, metrics, validation-selected pair threshold,
simple-fusion weight, and exact Top-20 protocol were SHA-256 locked before this one-time test run.
No checkpoint, weight, threshold, or hyperparameter was selected on test.

- Checkpoint SHA-256: `95289d84fbb85f99764f42b05ded92ec2c535b2b421b3fa1422cfb987b2800f4`
- Training config SHA-256: `279c96794c207fb2e62e4638cdae315dc4ffd4a2b85ecf039f41861e7412377c`
- Training metrics SHA-256: `da4d681cf0fb656905748a85c88f1285016ae4aee690826fdcd809521de4a313`
- Frozen pair threshold: `0.244170547`
- Frozen simple-fusion image weight: `0.40`

## Validation-to-test comparison

| Method | Validation mAP@20 | Test mAP@20 | Validation Recall@20 | Test Recall@20 |
|---|---:|---:|---:|---:|
| Image only | 0.53907 | 0.55675 | 0.64667 | 0.65941 |
| Text only | 0.75693 | 0.74841 | 0.87414 | 0.86978 |
| Simple score fusion | 0.87358 | 0.86277 | 0.94005 | 0.93098 |
| Learned fusion | 0.87023 | 0.85946 | 0.93780 | 0.93235 |
| Pair-head rerank | 0.87903 | 0.86848 | 0.93780 | 0.93235 |

## Pair decision at frozen threshold

| Metric | Test value |
|---|---:|
| Precision | 0.74840 |
| Recall | 0.63029 |
| F1 | 0.68429 |

## Efficiency

| Metric | Value |
|---|---:|
| Image extraction throughput | 174.86 listings/s |
| Text extraction throughput | 14345.76 listings/s |
| Fusion throughput | 77691.86 listings/s |
| Exact ranking p50 / p95 | 0.366 / 0.614 ms/query |

## Interpretation

The pair-head result is the primary learned Phase 5 output because it was the frozen checkpoint
target. mAP@20 measures whether true duplicates are ranked early and completely within Top-20;
Recall@20 measures the candidate ceiling but does not measure false-candidate volume. Pair F1 uses
the validation-frozen decision threshold and therefore assesses the match/no-match operating point
without test-time tuning.
