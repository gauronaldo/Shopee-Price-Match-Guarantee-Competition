# Entity Resolution Benchmark

This report consolidates validation-only graph development and the selected final inference
policy. Ground-truth labels are used for selection and analysis, never as graph features or
edge-construction inputs.

## Frozen inputs

- Listings: `3,430` validation listings
- Candidate budget: Top-`50` exact cosine neighbours
- Pair scorer: selected symmetric pair head
- Candidate Recall@50 ceiling: `0.97438`

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

This is a validation-selected operating point, not a final test claim.

Manual inspection shows two dominant categories: same-brand or same-package variants can still
form false-merge bridges, while large groups with diverse images and titles are fragmented by the
strict reciprocal and full cross-component-support rules. Some near-identical cross-label examples
also remain plausible label ambiguities and are documented rather than relabeled.

## Reproduction

```powershell
.venv\Scripts\shopee-entity-resolution benchmark `
  --config configs\experiment\entity_resolution_benchmark.yaml
```

## Validation recall-recovery experiments

The original frozen result above exposed a recall bottleneck: pairwise precision was high, but
large product groups were fragmented. Follow-up experiments retained the same group-disjoint
validation split, frozen pair scorer, and predeclared safety gates:

- pairwise precision at least `0.88`;
- pairwise recall at least `0.40` and pairwise F1 at least `0.55`;
- B-cubed F1 at least `0.84`;
- false-merge pair rate at most `0.11`;
- false-split group rate at most `0.30`.

| Validation policy | Pair P | Pair R | Pair F1 | B-cubed F1 | False merge | False split | Gate |
|---|---:|---:|---:|---:|---:|---:|---|
| Dense incumbent | 0.90165 | 0.33119 | 0.48444 | 0.82794 | 0.09835 | 0.30818 | fail |
| Supported singleton attachment | 0.89317 | 0.39144 | 0.54433 | 0.84616 | 0.10683 | 0.28091 | fail |
| Residual pair-evidence head | 0.92476 | 0.38122 | 0.53989 | 0.78442 | 0.07524 | 0.55000 | fail |
| Multi-source candidates + singleton attachment | **0.89582** | **0.45573** | **0.60413** | **0.85797** | **0.10418** | **0.26364** | **pass** |

### Supported singleton attachment

The graph first builds conservative core components. A second pass may attach a singleton only
when at least two distinct members of one established component support it. Core membership is
snapshotted before attachment, so a newly attached listing cannot bootstrap a transitive chain.
This reduced false splits but narrowly missed the pair-recall and pair-F1 gates.

### Pair-evidence ablation

An 11-parameter residual head combined the frozen pair probability with joint cosine,
pHash, train-fitted character TF-IDF, token overlap, digit/unit consistency, exact-title/pHash
flags, and title-length ratio. Candidate-pair average precision improved from `0.78044` to
`0.82450`, but no calibrated graph policy converted that gain into safe clustering quality. The
head is therefore documented as an unsuccessful ablation and is not part of the selected system.

### Selected final inference policy

Train-fitted character TF-IDF Top-50, frozen dense Top-50, and pHash Top-20 candidates are combined
with weighted reciprocal-rank fusion. Candidate Recall increased from dense Recall@50 `0.97438`
to final Recall@75 `0.99209`. The downstream pair scorer and all trained model weights remain
frozen; this is an inference-policy selection, not a newly trained model.

The selected graph uses a `0.14` core probability threshold, reciprocal rank `5`, full
cross-component coverage, and supported singleton attachment at probability `0.18` within rank
`50`. It produces 1,410 clusters, including 325 singletons; 189 supported singletons are attached.
All six validation gates pass.

The selected final system reaches test pairwise precision/recall/F1
`0.87850 / 0.40396 / 0.55344` and B-cubed F1 `0.84711`. False-split and false-merge rates are
`0.28350 / 0.12150`. Full evaluation details and interpretation are recorded in
[`final_evaluation.md`](final_evaluation.md).

```powershell
.venv\Scripts\shopee-entity-resolution recover-recall `
  --config configs\experiment\entity_recall_recovery.yaml
.venv\Scripts\shopee-entity-resolution train-pair-evidence `
  --config configs\experiment\pair_evidence_training.yaml
.venv\Scripts\shopee-entity-resolution select-pair-evidence-graph `
  --config configs\experiment\pair_evidence_graph_selection.yaml
.venv\Scripts\shopee-retrieval hybrid `
  --config configs\experiment\hybrid_candidate_retrieval.yaml
.venv\Scripts\shopee-entity-resolution evaluate-hybrid-candidates `
  --config configs\experiment\hybrid_entity_resolution.yaml
.venv\Scripts\shopee-final preflight-hybrid `
  --config configs\experiment\hybrid_system_evaluation.yaml
.venv\Scripts\shopee-final evaluate-hybrid `
  --config configs\experiment\hybrid_system_evaluation.yaml
```

EfficientNet-B1 fine-tuning is deferred because the selected multi-source retrieval policy resolves
the measured validation bottleneck without reopening encoder training or adding that compute cost.
