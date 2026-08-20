# Scratch image encoder frozen test evaluation

## Protocol

This report evaluates one SHA-256-locked scratch image checkpoint on the held-out test split.
The model, exact Top-20 retrieval protocol, and pair threshold were frozen
from validation before test labels were evaluated. No threshold or hyperparameter was selected on
test.

- Checkpoint SHA-256: `6ea26b493d643b148cbcc48006231637b266491a0a026d7fdbd22284f7100e07`
- Training config SHA-256: `7d9551060b2b47a023eaf00d39f92fbb2174102914101a613cfa2f8a1cb8c06a`
- Frozen validation mAP@20: `0.53907`
- Frozen validation pair threshold: `0.805664`

## Test result

| Metric | Value |
|---|---:|
| mAP@20 | 0.55674 |
| Recall@1 | 0.29743 |
| Recall@5 | 0.52854 |
| Recall@10 | 0.60340 |
| Recall@20 | 0.65941 |
| Pair precision | 0.83231 |
| Pair recall | 0.34693 |
| Pair F1 | 0.48973 |
| Embedding throughput (listings/s) | 159.93 |
| Ranking p50 / p95 (ms/query) | 0.450 / 0.717 |

## Evaluation policy

The frozen checkpoint was evaluated once. The test result was not used to select a checkpoint,
threshold, retrieval setting, or hyperparameter.
