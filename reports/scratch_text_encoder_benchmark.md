# Scratch text encoder benchmark

## Experiment status

This Phase 4 character TextCNN was initialized randomly. Its vocabulary was fitted from training
titles only; no pretrained tokenizer, word embedding, language model, image feature, or test label
entered training or checkpoint selection.

## Validation result

- Selected epoch: `26` of `30`
- mAP@20: `0.75698`
- Recall@1: `0.42089`
- Recall@5: `0.70905`
- Recall@10: `0.80204`
- Recall@20: `0.87414`
- Vocabulary size: `40`
- Validation unknown-character rate: `0.000000`
- Parameters: `455,040`

The Phase 2 character TF-IDF validation reference is mAP@20 `0.8635`. Smoke and bounded pilot
runs are engineering evidence, not final claims against that full baseline.

## Training curve

| Epoch | Train loss | Validation map@20 | Seconds |
|---:|---:|---:|---:|
| 0 | 0.58699 | 0.65048 | 5.72 |
| 1 | 0.40167 | 0.66567 | 5.81 |
| 2 | 0.29054 | 0.67684 | 5.37 |
| 3 | 0.21229 | 0.70350 | 5.31 |
| 4 | 0.15479 | 0.70473 | 5.23 |
| 5 | 0.13063 | 0.71132 | 5.00 |
| 6 | 0.09898 | 0.71375 | 5.55 |
| 7 | 0.08927 | 0.72050 | 5.20 |
| 8 | 0.07933 | 0.72461 | 5.55 |
| 9 | 0.07395 | 0.72992 | 5.73 |
| 10 | 0.05888 | 0.72874 | 5.29 |
| 11 | 0.06068 | 0.72661 | 5.53 |
| 12 | 0.04760 | 0.73445 | 5.19 |
| 13 | 0.04921 | 0.73514 | 5.05 |
| 14 | 0.04312 | 0.73784 | 5.82 |
| 15 | 0.04017 | 0.73986 | 5.71 |
| 16 | 0.03746 | 0.74127 | 5.77 |
| 17 | 0.03446 | 0.74133 | 5.66 |
| 18 | 0.03150 | 0.74458 | 5.05 |
| 19 | 0.02972 | 0.74529 | 5.23 |
| 20 | 0.02779 | 0.74749 | 5.01 |
| 21 | 0.02562 | 0.75032 | 5.92 |
| 22 | 0.02480 | 0.75149 | 5.96 |
| 23 | 0.02118 | 0.75097 | 5.19 |
| 24 | 0.02129 | 0.75480 | 5.56 |
| 25 | 0.02241 | 0.75698 | 5.02 |
| 26 | 0.02030 | 0.75607 | 5.66 |
| 27 | 0.01939 | 0.75669 | 5.56 |
| 28 | 0.02068 | 0.75687 | 5.28 |
| 29 | 0.02003 | 0.75679 | 5.78 |
