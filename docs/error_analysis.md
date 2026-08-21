# Error analysis

## Classical retrieval benchmark

The deterministic classical retrieval run uses split manifest
`c9cef390b5fbde6c833fddb15a0a8df2c7fbecacd8d50fb83aadba6056bf8e09`. Aggregate results are in
[`../reports/classical_retrieval.md`](../reports/classical_retrieval.md); raw
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
[`../reports/image_encoder.md`](../reports/image_encoder.md).

## Final scratch image review

The full 224-pixel checkpoint reached validation/test mAP@20 `0.53907 / 0.55674`. A deterministic
manual review categorized 20 top-1 false matches and 20 Top-20 retrieval misses. Shared layout,
background, color blocks, and coarse silhouette account for 70% of reviewed false matches and 75%
of reviewed retrieval misses. Coarse-category or brand-family negatives account for most remaining
cases; two false matches are probable label fragmentation.

Full counts, examples, sampling limitations, and the resulting Phase 4 motivation are recorded in
[`../reports/image_encoder.md`](../reports/image_encoder.md).

## Entity-resolution graph

The selected validation-only graph reaches pairwise precision `0.90165` and B-cubed F1 `0.82794`
under the configured false-merge safety gate. The bounded review artifact contains 91 impure
clusters, 339 split label groups, and 106 label-blind manual-review flags.

Manual inspection identifies three dominant categories:

- **Same-brand variant false merge:** matching packaging and quantity can bridge different flavors
  or variants, such as standard and extra-spicy products of the same brand.
- **Possible label fragmentation:** some cross-label clusters have near-identical titles, brand,
  and quantity. They remain false merges under competition ground truth and are not relabeled.
- **Large-group false split:** diverse images and wording inside groups of 10 or more listings do
  not provide full cross-component support, so the conservative graph produces several fragments.

The selected precision-first policy is therefore appropriate for avoiding catastrophic catalog
merges, but it does not solve recall for large heterogeneous entities. Detailed metrics and the
group-size breakdown are in [`../reports/entity_resolution.md`](../reports/entity_resolution.md).

## Final held-out system

The validation-frozen end-to-end policy reaches held-out pairwise precision/recall/F1
`0.89591 / 0.32723 / 0.47937` and B-cubed precision/recall/F1
`0.95279 / 0.72331 / 0.82234`. Relative to validation, B-cubed F1 decreases by `0.00559`, while
the false-merge pair rate increases from `0.09835` to `0.10409`. This small degradation supports
reasonable generalization but does not satisfy a strict 0.90 held-out pairwise-precision target.

The held-out review contains 95 impure clusters, 369 split label groups, and 91 manual-review
clusters. Fragmentation is strongly size-dependent:

- size-2 groups are recovered without splits 75.87% of the time;
- groups of size 3-5 are recovered without splits 61.54% of the time;
- groups of size 6-9 are recovered without splits 17.19% of the time;
- no group of size at least 10 is recovered as one entity; these groups average 7.69 fragments.

The highest-value future modeling work is therefore large-group recall and component linking under
strict false-merge controls, not a global threshold reduction. Lowering the threshold after seeing
test would violate the frozen protocol and is not performed.
