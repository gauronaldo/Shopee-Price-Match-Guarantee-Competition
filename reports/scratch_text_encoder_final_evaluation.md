# Scratch text encoder frozen test evaluation

## Protocol

One SHA-256-locked scratch TextCNN checkpoint is evaluated on the held-out test split. The
checkpoint, exact Top-20 protocol, and pair threshold were selected using
validation only.

- Checkpoint SHA-256: `cbb77e4d76c1909c24b0e30654eacde2a0f752bb5d2ba795d45b01ec1189f7c1`
- Training config SHA-256: `dbd9ac71a42f0dec92412c8aabbc3e987e71356ad6d8d0a01230eebcc93caf77`
- Training metrics SHA-256: `e92bd3704d09e6e3ae36e89c2d219c69e7b3d7daa6ad70e856ae0e0e73095030`
- Frozen validation map@20: `0.75698`
- Frozen validation pair threshold: `0.654422`

## Test result

| Metric | Value |
|---|---:|
| mAP@20 | 0.74841 |
| Recall@1 | 0.40737 |
| Recall@5 | 0.70742 |
| Recall@10 | 0.79967 |
| Recall@20 | 0.86978 |
| Pair precision | 0.65102 |
| Pair recall | 0.49420 |
| Pair F1 | 0.56187 |
| Embedding throughput (listings/s) | 10741.39 |
| Ranking p50 / p95 (ms/query) | 0.343 / 0.425 |

## Evaluation policy

This output is the single test evaluation for the frozen Phase 4 checkpoint. No test metric was
used to alter the model, threshold, vocabulary, sequence length, or retrieval protocol.
