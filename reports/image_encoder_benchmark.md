# Scratch image encoder benchmark

## Experiment status

This is an image-only Phase 3 run. The residual CNN and projection head were initialized randomly;
no pretrained weights, title features, pHash, or ORB scores entered the model. Checkpoint selection
used validation `map@20` only. This training command never accesses the test split; held-out test
metrics are produced separately by the frozen-checkpoint evaluator.

## Validation result

- Selected epoch: `39`
- mAP@20: `0.53907`
- Recall@1: `0.28599`
- Recall@5: `0.50456`
- Recall@10: `0.58195`
- Recall@20: `0.64667`
- Embedding throughput: `270.50` listings/second
- Parameters: `3,060,000`
- Serialized checkpoint: `36,876,216` bytes

Phase 2 reference points on the same real validation split are pHash mAP@20 `0.2895` and ORB
mAP@20 `0.6638`. A smoke or bounded pilot run is not a fair claim against those full-data
baselines.

## Training curve

| Epoch | Train loss | Validation map@20 | Seconds |
|---:|---:|---:|---:|
| 0 | 2.30263 | 0.31742 | 290.08 |
| 1 | 2.00578 | 0.34588 | 287.43 |
| 2 | 1.74569 | 0.37301 | 293.41 |
| 3 | 1.61756 | 0.39026 | 319.53 |
| 4 | 1.47636 | 0.41259 | 307.87 |
| 5 | 1.37675 | 0.42069 | 306.67 |
| 6 | 1.26441 | 0.44088 | 307.79 |
| 7 | 1.15067 | 0.45443 | 296.44 |
| 8 | 1.13491 | 0.45271 | 298.29 |
| 9 | 1.06701 | 0.46262 | 312.13 |
| 10 | 1.01459 | 0.47075 | 280.31 |
| 11 | 0.93965 | 0.48091 | 305.99 |
| 12 | 0.89139 | 0.48134 | 305.33 |
| 13 | 0.84866 | 0.48621 | 305.06 |
| 14 | 0.83585 | 0.49461 | 305.65 |
| 15 | 0.77609 | 0.49379 | 306.77 |
| 16 | 0.73450 | 0.49993 | 306.89 |
| 17 | 0.69145 | 0.50391 | 305.12 |
| 18 | 0.65562 | 0.50511 | 303.46 |
| 19 | 0.61493 | 0.50964 | 308.98 |
| 20 | 0.58305 | 0.51296 | 320.89 |
| 21 | 0.57296 | 0.51208 | 318.16 |
| 22 | 0.51700 | 0.51696 | 313.68 |
| 23 | 0.48940 | 0.52040 | 353.89 |
| 24 | 0.49389 | 0.52843 | 366.64 |
| 25 | 0.45374 | 0.52802 | 357.07 |
| 26 | 0.44045 | 0.52646 | 359.89 |
| 27 | 0.39332 | 0.52804 | 370.04 |
| 28 | 0.38958 | 0.52949 | 363.39 |
| 29 | 0.37180 | 0.53481 | 356.94 |
| 30 | 0.35540 | 0.52822 | 364.93 |
| 31 | 0.35514 | 0.53165 | 359.06 |
| 32 | 0.34718 | 0.53489 | 356.38 |
| 33 | 0.31264 | 0.53591 | 361.44 |
| 34 | 0.32466 | 0.53592 | 432.92 |
| 35 | 0.32759 | 0.53517 | 356.47 |
| 36 | 0.30967 | 0.53595 | 357.24 |
| 37 | 0.31182 | 0.53528 | 355.58 |
| 38 | 0.30172 | 0.53804 | 358.33 |
| 39 | 0.30881 | 0.53907 | 354.92 |

## Reproducibility

- Seed: `2026`
- Git commit: `3c4b54de7a36ae17673e070763f5ac590a3ae98c`
- Split manifest SHA-256: `c9cef390b5fbde6c833fddb15a0a8df2c7fbecacd8d50fb83aadba6056bf8e09`
- Device: `cuda`
- Initialization: `kaiming_normal_conv_linear; batch_norm_unit_scale; random_only`
- Normalization: `rgb_[0,1]_then_(x-0.5)/0.5; fixed_non_pretrained`
