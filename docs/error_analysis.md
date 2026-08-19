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
- **Classical pair matcher:** train-only Logistic Regression improves test mAP@20 from 0.8810 to
  0.9004 and pair F1 from 0.7220 to 0.7416. Its quantity-conflict coefficient is negative as
  intended, while correlated features such as exact title and pHash require conditional rather
  than causal interpretation.
- **Candidate ceiling:** the label-blind pHash/TF-IDF union reaches 0.9512 macro positive recall on
  test. The pair matcher retains 0.9467 in its final Top 20, leaving limited ranking headroom before
  candidate generation itself must improve.
- **Efficiency trade-off:** pair matching reaches about 83 ms/query versus 17 ms/query for weighted
  fusion in the development run because ORB remains CPU-heavy. The quality gain is measurable but
  does not justify treating ORB as the future production representation.

These findings motivate a scratch image encoder that learns product-level evidence rather than
depending on pHash or local keypoints. Later phases will retain the taxonomy for retrieval misses,
pair-score errors, transitive false merges, and false splits. No private or restricted image is
copied into Git.
