# Entity Resolution Benchmark

Phase 8 status: **phase8_complete_validation_only**. Thresholds and graph rules are selected on validation only;
test remains untouched. Ground-truth labels are used for selection and analysis, never as graph
features or edge-construction inputs.

## Frozen inputs

- Listings: `3,430` validation listings
- Candidate budget: Top-`50` exact cosine neighbours
- Pair scorer: accepted Phase 6 symmetric pair head
- Candidate Recall@50 ceiling: `0.97438`
- Test accessed: `false`

## Selected graph policy

| Setting | Selected value |
|---|---:|
| Pair probability threshold | 0.160000 |
| Reciprocal-neighbour rank | 5 |
| Cross-component coverage | 1.00 |
| Variant-conflict override probability | 0.15 |
| Maximum cluster size | 64 |

## Validation metrics

| Metric family | Precision | Recall | F1 |
|---|---:|---:|---:|
| Accepted candidate edges | 0.81339 | 0.38693 | 0.52440 |
| Pairwise clusters | 0.90165 | 0.33119 | 0.48444 |
| B-cubed clustering | 0.95618 | 0.73003 | 0.82794 |

- False-merge pair rate: `0.09835`
- Impure non-singleton cluster rate: `0.08529`
- False-split group rate: `0.30818`
- Predicted clusters / singleton clusters: `1605` / `538`
- Maximum predicted cluster size: `7`
- Manual-review clusters: `106`

## Graph audit

| Counter | Value |
|---|---:|
| Unique Top-K candidate pairs | 106,240 |
| Eligible reciprocal edges | 4,003 |
| Accepted component merges | 1,825 |
| Rejected by cluster-size cap | 0 |
| Rejected by transitive consistency | 1,006 |
| Rejected variant conflicts | 0 |

## Performance by true group size

| Group size | Groups | Unsplit-group rate | Mean predicted fragments |
|---|---:|---:|---:|
| 2 | 695 | 0.78705 | 1.213 |
| 3_to_5 | 311 | 0.65273 | 1.418 |
| 6_to_9 | 65 | 0.16923 | 2.738 |
| 10_plus | 29 | 0.00000 | 8.103 |

## Interpretation

Pairwise precision is the primary false-merge safety metric because one false edge can merge
otherwise correct components. B-cubed F1 balances entity purity and fragmentation per listing.
The reciprocal-neighbour rule removes one-sided retrieval coincidences; the cross-component
coverage rule blocks a single bridge from joining two established components unless enough members
support the merge. Variant-conflicting titles require a higher pair probability.

This is a validation-selected operating point, not a final test claim. Detailed false-merge,
false-split, and manual-review examples remain in the ignored Phase 8 review artifact.

Manual inspection shows two dominant categories: same-brand or same-package variants can still
form false-merge bridges, while large groups with diverse images and titles are fragmented by the
strict reciprocal and full cross-component-support rules. Some near-identical cross-label examples
also remain plausible label ambiguities and are documented rather than relabeled.

## Reproduction

```powershell
.venv\Scripts\shopee-entity-resolution benchmark `
  --config configs\experiment\entity_resolution_benchmark.yaml
```
