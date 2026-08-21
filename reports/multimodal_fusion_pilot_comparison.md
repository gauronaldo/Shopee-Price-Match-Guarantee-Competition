# Multimodal fusion pilot comparison

## Decision

The Phase 5 implementation and pilot gates pass. Frozen-embedding
caching, simple score fusion, projected learned fusion, score-preserving residual fusion, and a
symmetric pair head all run reproducibly on train/validation without accessing test.

The initial residual pair-head pilot motivated a controlled pair-loss ablation. Weight `0.10`
subsequently improved validation mAP@20 to `0.87903` and pair F1 to `0.71285`, so it was selected
for the full run documented in
[`multimodal_fusion_training_summary.md`](multimodal_fusion_training_summary.md).

## Validation results

| Method | mAP@20 | Recall@20 | Pair F1 at selected threshold |
|---|---:|---:|---:|
| Frozen image only | 0.53907 | 0.64667 | not reselected |
| Frozen text only | 0.75693 | 0.87414 | not reselected |
| Simple score fusion, image weight 0.40 | 0.87358 | **0.94005** | **0.70437** |
| Projected learned fusion | 0.81244 | 0.90817 | 0.65916 |
| Projected fusion + pair-head rerank | 0.80868 | 0.90817 | 0.65554 |
| Residual learned fusion, selected epoch | 0.86651 | 0.93666 | 0.68999 |
| Residual fusion + pair-head rerank | **0.87565** | 0.93666 | 0.70274 |

The random projected model improves over frozen text by `+0.05551` mAP@20, proving that the
learned fusion path is functional, but it discards too much of the strong lexical baseline. The
residual model begins from the exact score-fusion geometry and learns only a correction. Its pair
head improves the selected candidate ordering enough to exceed simple fusion mAP, while the joint
embedding alone remains below the simple baseline.

## Training behavior

The residual pair-head system selected epoch 1. Pair-head mAP then fell from `0.87565` to
`0.85479`, `0.85113`, `0.84907`, and `0.84998`, even while total training loss continued to fall.
This is evidence that the current `1.0` contrastive plus `0.5` pair-BCE objective is not aligned
perfectly with validation ranking quality. More epochs with the same setup are therefore unlikely
to be the highest-value next step.

## Efficiency and reproducibility

- Frozen cache: 27,391 train and 3,430 validation listings; test excluded.
- Train cache size: approximately 52.3 MB; validation cache size: approximately 6.5 MB.
- One-time frozen image extraction: approximately 247 seconds across train and validation.
- Frozen text extraction: approximately 2.5 seconds.
- Residual fusion trainable parameters: 1,051,137.
- Residual checkpoint size: approximately 12.6 MB.
- Frozen encoder SHA-256 values and preprocessing/tokenization contracts are stored with each cache.
- Cached validation mAP differs from the source reports by only `1.43e-6` for image and `4.49e-5`
  for text, inside the explicit `1e-4` float32 reproduction tolerance.

## Gate resolution

The validation-only ablation changed one factor at a time:

1. pair-BCE `0.50`: mAP@20 `0.87565`, pair F1 `0.70274`;
2. pair-BCE `0.25`: mAP@20 `0.87868`, pair F1 `0.71081`;
3. pair-BCE `0.10`: mAP@20 `0.87903`, pair F1 `0.71285`.

Weight `0.10` leads both selection metrics and was used for full training. The held-out test split
remains locked.
