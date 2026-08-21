# Final System Evaluation

Status: **final_system_test_complete**. The complete custom system was evaluated once with the exact
validation-selected retrieval, pair-scoring, and entity-resolution policy. No parameter,
threshold, candidate K, or graph rule was selected on this test result.

## Frozen contract

- Source commit: `f87639b8942020cbd0ba04a2113f3edb15f0d3d3` (`git_dirty=false`)
- Final config SHA-256: `2f7741c3ec5a5e7032731029c2842f2219aae2a0e6b81d59eb5875fcc5d78d44`
- Entity config SHA-256: `84b68e8478a237553e27cf41296ec9f47a1a146185d5657402e1330608a4c794`
- Entity metrics SHA-256: `1d8c65a14d9cb9a4927bd3d0f56f7a7e2f7eab5e85f1a493bb856aa60b34fe1f`
- Phase 6 checkpoint SHA-256: `d763834919c9bea2378b112e870d15b82817023692940c20f112f98d49370c3e`
- Split manifest SHA-256: `c9cef390b5fbde6c833fddb15a0a8df2c7fbecacd8d50fb83aadba6056bf8e09`
- Candidate K / pair threshold / reciprocal rank: `50` /
  `0.16` / `5`
- Cross-component coverage / maximum cluster size:
  `1.00` / `64`

## Retrieval: validation to test

| Metric | Validation | Test |
|---|---:|---:|
| mAP@20 | 0.87023 | 0.85946 |
| Recall@20 | 0.93780 | 0.93235 |
| mAP@50 | 0.87279 | 0.86001 |
| Recall@50 | 0.97438 | 0.96882 |

## Pair decisions on retrieved candidates

| Metric | Test value |
|---|---:|
| Raw pair-head precision | 0.68519 |
| Raw pair-head recall within candidates | 0.76341 |
| Raw pair-head F1 within candidates | 0.72219 |
| Average precision / PR-AUC | 0.78497 |
| Brier score | 0.04992 |
| Expected calibration error | 0.08596 |
| Accepted reciprocal-edge precision | 0.81962 |
| Accepted reciprocal-edge global recall | 0.38345 |
| Accepted reciprocal-edge F1 | 0.52247 |

Raw pair-head metrics are candidate-conditioned. Accepted-edge recall uses every true test pair as
its denominator and therefore includes retrieval and reciprocal-gating misses.

## Entity resolution: validation to test

| Metric | Validation | Test |
|---|---:|---:|
| Pairwise precision | 0.90165 | 0.89591 |
| Pairwise recall | 0.33119 | 0.32723 |
| Pairwise F1 | 0.48444 | 0.47937 |
| B-cubed precision | 0.95618 | 0.95279 |
| B-cubed recall | 0.73003 | 0.72331 |
| B-cubed F1 | 0.82794 | 0.82234 |
| False-merge pair rate | 0.09835 | 0.10409 |
| False-split group rate | 0.30818 | 0.33637 |

## Efficiency

| Stage | Measured result |
|---|---:|
| Image extraction | 188.46 listings/s |
| Text extraction | 15959.47 listings/s |
| Joint fusion | 83475.14 listings/s |
| Pair scoring | 40063.06 pairs/s |
| Exact query p50 / p95 | 0.344 /
  0.416 ms |
| End-to-end wall time | 23.88 s |

## Interpretation and disclosure

This is the first evaluation of the complete retrieval-plus-pair-plus-clustering system on the
held-out split. Earlier phases already reported component-level image, text, and multimodal test
results on the same frozen split; therefore it is held out from system-policy selection, but it is
not globally unseen to the project owner. The final test result is descriptive and is not used to
revise the operating point.

Detailed false-merge, false-split, and review examples remain in the ignored local artifact.
Aggregate failure counts and group-size strata are retained in the metrics JSON.

## Reproduction guard

```powershell
.venv\Scripts\shopee-final preflight `
  --config configs\experiment\final_system_evaluation.yaml
.venv\Scripts\shopee-final evaluate `
  --config configs\experiment\final_system_evaluation.yaml
```

The access marker and immutable outputs intentionally block a second evaluation.
