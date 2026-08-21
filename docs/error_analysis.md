# Error analysis

## Classical retrieval benchmark

The deterministic classical retrieval run uses split manifest
`c9cef390b5fbde6c833fddb15a0a8df2c7fbecacd8d50fb83aadba6056bf8e09`. Aggregate results are in
[`../reports/classical_retrieval_benchmark.md`](../reports/classical_retrieval_benchmark.md); raw
titles and listing IDs for the bounded review sample remain in ignored
`artifacts/classical_retrieval/review_examples.json`.

Manual inspection of five top-ranked failures per baseline found:

- **pHash collision / retrieval miss:** visually coarse 64-bit similarity can rank semantically
  unrelated categories together and has low Recall@20.
- **Variant or model ambiguity:** TF-IDF and fusion can prefer the same product family when titles
  omit or de-emphasize identity-critical quantities, model numbers, or bundle details.
- **Shared visual structure:** ORB can favor repeated packaging layouts, logos, and object geometry
  even when the purchasable product differs.
- **Probable label fragmentation:** several highly similar cross-label titles appear plausibly to
  describe the same exact item. They remain errors against competition ground truth and support
  the Phase 1 warning that measured precision may understate commercial matching quality.
- **Modality disagreement:** the selected fusion assigns 75% weight to text. Image evidence helps
  aggregate retrieval, but weak pHash evidence can still perturb otherwise strong title rankings.

These findings motivate a scratch image encoder that learns product-level evidence rather than
depending on pHash or local keypoints. Later phases will retain the taxonomy for retrieval misses,
pair-score errors, transitive false merges, and false splits. No private or restricted image is
copied into Git.

## Scratch image encoder pilot

The selected bounded pilot used `P=16, K=2` product-aware batches and reached validation mAP@20
`0.34738`. Its exact-positive-pHash and no-exact-positive-pHash strata reached `0.72553` and
`0.21866`, respectively. The latter improved by 29.6% relative to the otherwise matched `P=8`
pilot, which supports a genuine improvement in learned visual retrieval rather than only recovery
of exact duplicates.

A manual image-only review of the first ten top-1 false matches found that global visual layout
was the dominant error source. Similar white or brown backgrounds, centered dark objects,
rectangular packages, advertisement panels, and dense catalog collages caused matches between
unrelated product categories. Two cases were coarse-category hard negatives: different bags and
different instant-noodle products. These examples show that the model captures category and
composition before it consistently captures exact-product identity.

The next approved experiment is the already configured 224-pixel full run. Higher resolution can
preserve packaging text, logos, and fine product details that are suppressed at 128 pixels. This
is a targeted response to the measured layout-shortcut failure; no larger backbone or additional
loss is introduced before that hypothesis is evaluated. Quantitative details are recorded in
[`../reports/image_encoder_pilot.md`](../reports/image_encoder_pilot.md).

## Final scratch image review

The full 224-pixel checkpoint reached validation/test mAP@20 `0.53907 / 0.55674`. A deterministic
manual review categorized 20 top-1 false matches and 20 Top-20 retrieval misses. Shared layout,
background, color blocks, and coarse silhouette account for 70% of reviewed false matches and 75%
of reviewed retrieval misses. Coarse-category or brand-family negatives account for most remaining
cases; two false matches are probable label fragmentation.

Full counts, examples, sampling limitations, and the resulting Phase 4 motivation are recorded in
[`../reports/image_encoder_failure_analysis.md`](../reports/image_encoder_failure_analysis.md).
