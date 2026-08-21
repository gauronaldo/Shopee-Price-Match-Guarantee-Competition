# Multimodal fusion repeated-seed validation summary

## Protocol

Seeds 2026, 2027, and 2028 use the same leakage-safe split, frozen image/text checkpoints,
residual fusion architecture, loss weights, batch construction, early-stopping rule, and full
validation candidate pool. Only initialization and deterministic seed-controlled sampling differ.
The original seed 2026 remains the canonical checkpoint; the best observed seed is not selected
retroactively.

## Results

| Seed | Best epoch | Pair-head mAP@20 | Recall@1 | Recall@5 | Recall@10 | Recall@20 |
|---:|---:|---:|---:|---:|---:|---:|
| 2026 | 1 | 0.87903 | 0.48957 | 0.80954 | 0.88735 | 0.93780 |
| 2027 | 1 | 0.88157 | 0.49236 | 0.80884 | 0.88552 | 0.93841 |
| 2028 | 1 | 0.87965 | 0.49087 | 0.80525 | 0.88627 | 0.93894 |
| Mean | - | **0.88008** | **0.49094** | **0.80787** | **0.88638** | **0.93838** |
| Sample standard deviation | - | 0.00132 | 0.00139 | 0.00230 | 0.00092 | 0.00057 |

The standard deviation is small relative to the mean, so the validation result is not explained by
one lucky initialization. Mean pair-head mAP@20 is `+0.00651` above the deterministic simple-score
fusion reference (`0.87358`). Recall@20 is essentially flat because pair-head reranking operates
inside the learned fusion's fixed Top-20 candidate set; its primary contribution is ordering quality.

All three runs peak after the first epoch and then reduce training loss while validation mAP falls.
This is consistent evidence of rapid overfitting in the small trainable fusion/pair head and supports
the existing early-stopping policy. It also motivates Phase 6 hard-negative mining more strongly than
simply increasing Phase 5 training duration.
