# Scratch text encoder pilot benchmark

## Decision

The bounded Phase 4 pilot is complete on the frozen validation split. The train-only character
TextCNN improved steadily through its final configured epoch, so the implementation is stable and
the architecture is promoted unchanged to full training. Test remains locked.

## Controlled run

The pilot used seed `2026`, 128-character inputs, random initialization, `P=16, K=2`, 100 batches
per epoch, 12 epochs, supervised contrastive loss, and exact full-split cosine validation
retrieval. It ran on the NVIDIA RTX 4060 in approximately 34 seconds.

| Metric | Smoke | Pilot |
|---|---:|---:|
| Validation mAP@20 | 0.64049 | **0.71376** |
| Recall@1 | 0.37372 | **0.40375** |
| Recall@5 | 0.61106 | **0.66917** |
| Recall@10 | 0.67851 | **0.75912** |
| Recall@20 | 0.73099 | **0.82911** |
| Validation pair F1 | 0.55890 | **0.56102** |

Pilot mAP@20 improves over smoke by `0.07327`. The selected threshold `0.635659` gives validation
pair precision `0.65704` and recall `0.48948`. Positive-pair cosine averages `0.63948`, while the
bounded negative sample averages `0.00526`, showing useful separation without embedding collapse.

## Stratified result

| Stratum | Validation mAP@20 |
|---|---:|
| Group size 2 | 0.73358 |
| Group size 3–5 | 0.70753 |
| Group size 6–9 | 0.65653 |
| Group size 10+ | 0.72549 |
| Normalized title length 0–30 | 0.68403 |
| Normalized title length 31–60 | 0.74043 |
| Normalized title length 61–100 | 0.70044 |
| Normalized title length 101+ | 0.58047 |

Long titles remain weakest, although the 101+ band improved from smoke `0.47240` to `0.58047`.
The next error analysis should check whether relevant identity tokens occur after the 128-character
cutoff or whether long titles primarily contain distracting seller keywords.

## Efficiency

- Parameters: `455,040`.
- Checkpoint size: `5,487,715` bytes.
- Validation embedding throughput: `11,292.71 listings/s`.
- Exact ranking p50 / p95: `0.348 / 0.447 ms/query`.
- Wall time: `33.59 seconds`.

## Full-training gate

The best checkpoint occurs at epoch 11 (zero-indexed), and validation mAP@20 still rises at the
final epoch. The full configuration therefore increases training exposure while preserving the
architecture, vocabulary policy, sampler composition, loss, and selection metric. The approved
next command is `shopee-text train --config configs/experiment/text_embedding_training.yaml`.

The Phase 2 TF-IDF validation mAP@20 remains `0.8635`; the pilot gap is `0.1497`. Full training
must be evaluated honestly against that strong non-neural baseline before Phase 4 can close.

