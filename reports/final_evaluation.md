# Frozen System Evaluation

The project ships one selected final system. Its image encoder, text encoder, multimodal fusion,
and pair head use the same frozen checkpoint established before the recall-recovery experiments.
Those experiments changed candidate generation and graph inference, not the trained model weights.

| Selected final-system metric | Validation | Test |
|---|---:|---:|
| Retrieval Recall@75 | 0.99209 | 0.98615 |
| Pairwise precision | 0.89582 | 0.87850 |
| Pairwise recall | 0.45573 | 0.40396 |
| Pairwise F1 | 0.60413 | 0.55344 |
| B-cubed F1 | 0.85797 | 0.84711 |
| False-merge pair rate | 0.10418 | 0.12150 |
| False-split group rate | 0.26364 | 0.28350 |

## Selected final system evaluation

Status: **complete**. The immutable artifact records the internal status
`hybrid_system_confirmatory_test_complete`.

### Frozen contract

- Source commit: `51d109d626d444ca9b8e4dfd423a9459c08f6346` (`git_dirty=false`)
- Final config SHA-256: `9f3c88a8e154101882821815681e2ce94cc3f2a3b9e0f5681ce0412022618957`
- Hybrid entity config SHA-256: `982d6118bfde3b444d672a0b5717dc4833097bf0e0fd0fd09a3e53158cad083e`
- Hybrid entity metrics SHA-256: `8833c0f6994b98dd08f10a4aa482d46e9520c25612b6b3e4be9ed6cd11a1472e`
- Confirmatory metrics SHA-256: `f4f30ae0158acc73e1563ac4bef1d9e97c89ccff4b78324ae2182ed2d6738841`
- Candidate K / core threshold / reciprocal rank: `75 / 0.14 / 5`
- Singleton threshold / reciprocal rank / minimum support: `0.18 / 50 / 2`
- Test-time parameter selection: disabled

### Candidate retrieval

| Metric | Validation | Test |
|---|---:|---:|
| mAP@20 | 0.90131 | 0.89245 |
| Recall@20 | 0.95929 | 0.95371 |
| mAP@50 | 0.90249 | 0.89164 |
| Recall@50 | 0.98962 | 0.98048 |
| mAP@75 | 0.90333 | 0.89364 |
| Recall@75 | 0.99209 | 0.98615 |

Recall is macro-averaged over queries. An independent validation micro average over directed
positive pairs was `0.97071`, which is lower because large groups receive more weight.

### Retrieval-integrity audit

| Invariant | Validation | Test |
|---|---:|---:|
| Query appears in its own candidates | 0 | 0 |
| Queries with duplicate candidate IDs | 0 | 0 |
| Queries above the Top-75 cap | 0 | 0 |
| Candidate count, minimum / maximum | 53 / 75 | 61 / 75 |
| Singleton queries | 0 | 0 |
| Unknown candidate IDs | 0 | 0 |

The metric raises an error when a query has no other positive; it never assigns singleton recall
of one. `CorpusItem`, the object exposed to candidate retrievers, contains only `posting_id`,
`image`, `image_phash`, and `title`, so `label_group` remains evaluation-only. Character TF-IDF is
fitted from train items and then applied to validation or test. The split audit found zero shared
label groups across train/validation, train/test, and validation/test.

Candidate retrieval applies no score threshold. Dense and character TF-IDF each return Top-50;
pHash returns Top-20. Every source removes self before its own Top-K cut. RRF defensively removes
self again, merges and deduplicates the three lists, then truncates the union to Top-75. Since no
single source supplies 75 candidates, strong overlap can leave fewer than 75 unique IDs. Pair
thresholds `0.14` and `0.18` are applied only after retrieval.

### Pair decisions

| Metric | Test value |
|---|---:|
| Candidate-conditioned precision | 0.58916 |
| Candidate-conditioned recall | 0.79964 |
| Candidate-conditioned F1 | 0.67845 |
| PR-AUC | 0.76610 |
| Brier score | 0.03351 |
| Expected calibration error | 0.06757 |
| Accepted-edge precision | 0.83803 |
| Accepted-edge global recall | 0.38623 |
| Accepted-edge F1 | 0.52876 |

Candidate-conditioned metrics score only retrieved pairs. Accepted-edge recall uses every true
test pair as its denominator and therefore includes retrieval and graph-gating misses.

### Entity resolution

| Metric | Validation | Test |
|---|---:|---:|
| Pairwise precision | 0.89582 | 0.87850 |
| Pairwise recall | 0.45573 | 0.40396 |
| Pairwise F1 | 0.60413 | 0.55344 |
| B-cubed precision | 0.95148 | 0.94269 |
| B-cubed recall | 0.78119 | 0.76913 |
| B-cubed F1 | 0.85797 | 0.84711 |
| False-merge pair rate | 0.10418 | 0.12150 |
| False-split group rate | 0.26364 | 0.28350 |

The test result remains directionally consistent with validation, but the validation safety gate
does not transfer perfectly: pairwise precision falls below `0.88` by `0.00150`, and false-merge
rate exceeds `0.11` by `0.01150`. Pairwise recall, pairwise F1, B-cubed F1, and false-split rate
retain the intended improvement established during development.

### Efficiency

| Stage | Test result |
|---|---:|
| Image extraction | 19.73 s |
| Text extraction | 0.30 s |
| Joint fusion | 0.04 s |
| TF-IDF fit and ranking | 29.66 s |
| pHash ranking | 4.04 s |
| Rank fusion | 0.38 s |
| Pair scoring | 37,000 pairs/s |
| Dense query p50 / p95 | 0.323 / 0.453 ms |
| End-to-end wall time | 62.05 s |

The exact dense stage remains fast; train-fit sparse ranking dominates this offline evaluation.
Serving should persist the fitted vocabulary, IDF values, and catalog index instead of fitting them
per request.

## Evaluation disclosure

The final inference policy was selected from 248 predeclared validation graph configurations. That
search creates some risk of validation optimism, so the confirmatory test is the stronger evidence
for its effect size. The same test split had already been used for earlier component experiments;
the final result is therefore not described as globally unseen. No policy was
revised after observing the test output.

The local access marker and immutable artifact path block accidental repetition of the final
evaluation.
