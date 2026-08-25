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

## Component-recovery experiment

A follow-up validation-only experiment keeps the encoders, fusion module, pair head, candidate
ranking, core graph, and supported-singleton policy frozen. It adds one label-blind pass that can
attach a component of two or three listings to a larger component when multiple distinct members
support the same target. Component membership is snapshotted before the pass, preventing newly
attached fragments from creating transitive evidence.

| Validation metric | Selected final policy | Component recovery | Delta |
|---|---:|---:|---:|
| Pairwise precision | 0.89582 | 0.89405 | -0.00177 |
| Pairwise recall | 0.45573 | **0.53048** | **+0.07475** |
| Pairwise F1 | 0.60413 | **0.66587** | **+0.06174** |
| B-cubed F1 | 0.85797 | **0.86953** | **+0.01156** |
| False-merge pair rate | 0.10418 | 0.10595 | +0.00177 |
| False-split group rate | 0.26364 | 0.26182 | -0.00182 |

The selected recovery policy attaches 34 fragments. It requires full support from every source
member, support from at least two target members, no digit/unit variant conflict, probability at
least `0.14`, and reciprocal rank at most `20`. It remains within the established precision and
false-merge safety limits.

The pairwise metrics above are standard clustering metrics over every pair assigned to the same
predicted cluster; they are not direct-edge metrics. Component recovery adds 712 same-cluster
pairs on validation: 629 true and 83 false, for recovery-induced pair precision `0.88343`. The 34
component merges only make two additional true groups completely unsplit. Other affected groups
remain partially fragmented, explaining why pairwise recall rises much more than the binary
group-level false-split rate improves.

The exploratory target required a false-split rate at most `0.26000`; the run reaches `0.26182`, so
it is recorded as a promising validation result rather than replacing the frozen final policy.
Relaxing source coverage to two of three members raises F1 slightly to `0.66830` but exceeds the
false-merge limit, so that alternative is rejected.

Most remaining false splits are singleton patterns. Directly linking singleton pairs at broader
thresholds produced only `0.67–0.75` validation precision, which is too risky for automatic merge.

### Hard-positive pair-head ablation

The follow-up training experiment carved a fresh 15% group-disjoint holdout from the original
train partition. It froze the image encoder, text encoder, and fusion module, then fine-tuned only
the symmetric pair head using 20,000 low-scoring true pairs, 17,310 mined hard negatives, random
positives, and random negatives. Test data was disabled.

| Metric | Before | Fine-tuned pair head | Delta |
|---|---:|---:|---:|
| Train-holdout pair F1 at precision ≥ 0.88 | 0.76697 | 0.76924 | +0.00227 |
| Validation graph pairwise precision | 0.89405 | 0.73727 | -0.15678 |
| Validation graph pairwise recall | 0.53048 | 0.58491 | +0.05443 |
| Validation graph pairwise F1 | 0.66587 | 0.65231 | -0.01356 |
| Validation graph false-merge rate | 0.10595 | 0.26273 | +0.15678 |

The checkpoint passed its internal train-holdout selection but failed every graph-level acceptance
gate. It gained recall by making pair probabilities broadly more permissive, causing unsafe
component merges rather than learning a better precision–recall boundary. The checkpoint is
rejected and does not replace the component-recovery policy. This result also shows why pair-level
holdout F1 alone is insufficient for selecting an entity-resolution system.

### Component-recovery confirmatory test

After a clean validation rerun at commit `fee6e58`, the exact validation-selected policy was locked
and applied once to the predecessor system's immutable Top-75 test candidate pairs. Candidate
generation and pair scores were reused unchanged, `label_group` was available only to the metric
functions, and no test-time selection was permitted.

| Test metric | Frozen predecessor | Component recovery | Delta |
|---|---:|---:|---:|
| Pairwise precision | **0.87850** | 0.85505 | -0.02345 |
| Pairwise recall | 0.40396 | **0.43283** | +0.02886 |
| Pairwise F1 | 0.55344 | **0.57473** | +0.02129 |
| B-cubed F1 | 0.84711 | **0.85097** | +0.00387 |
| False-merge pair rate | **0.12150** | 0.14495 | +0.02345 |
| False-split group rate | 0.28350 | **0.27621** | -0.00729 |

The test run makes 28 component merges and adds 400 same-cluster pairs: 249 true and 151 false.
Recovery-induced pair precision falls from `0.88343` on validation to `0.62250` on test. Although
recall, F1, B-cubed F1, and false splits improve, the precision and false-merge degradation fails
the catalog-safety objective. Component recovery is therefore retained as a documented ablation
and does not replace the frozen predecessor policy.

### Development protocol v2

Further recall work starts from a new group-disjoint protocol rather than tuning against the
historical test result. The original test membership is copied unchanged into an unavailable
`historical_test` role. All remaining leakage super-components are deterministically reassigned:

| Protocol role | Listings | Label groups | Usage |
|---|---:|---:|---|
| Train | 24,649 | 7,939 | Fit all v2 model parameters and train-only statistics |
| Development | 3,086 | 986 | Select architecture, thresholds, and graph policy |
| Confirmation | 3,086 | 992 | One-time evaluation after development is frozen |
| Historical test | 3,429 | 1,097 | Preserve prior evidence; unavailable for v2 selection |

Integrity checks report zero `label_group` overlap, zero leakage-super-component overlap, and an
exact match between old test IDs and the preserved historical-test IDs. Because the old encoders
were trained with a different group assignment, they cannot be presented as leakage-free v2
models. Any v2 learned representation or verifier must be trained under this new manifest before
confirmation is accessed.

### Pair-recall refinement experiments

An initial pilot on the preceding development protocol fine-tuned only the fusion module and
symmetric pair head. Its best checkpoint increased development recall from `0.40898` to `0.44152`,
but precision fell to `0.80152`, F1 improved by only `0.00850`, and false merge rose to `0.19848`.
It failed the locked graph-level
safety gates; the checkpoint was rejected and the result is retained only as an ablation.

The full-joint cycle then adopted a narrower protocol without reopening the prior confirmation or
historical-test partitions. The existing training partition remains unchanged; the prior
development pool is divided group-disjointly into a 1,542-listing development split and a
1,544-listing internal-confirmation split. Internal-confirmation labels are not loaded by training
or development evaluation. This is internal evidence rather than a new external test set.

The full-joint experiment fine-tuned the image encoder, text encoder, fusion module, and pair
head with tiered learning rates. Batches combined hard positives, hard negatives, random positives,
and random negatives. Distillation on negative pairs penalized large deviations from the frozen
pair scorer. Candidate generation returned at most 75 unique candidates per query, and TF-IDF
was fit on the unchanged training partition only.

| Development-v3 metric | Frozen system | Full joint, epoch 1 | Delta |
|---|---:|---:|---:|
| Pairwise precision | **0.91770** | 0.88403 | -0.03366 |
| Pairwise recall | 0.40800 | **0.42838** | **+0.02039** |
| Pairwise F1 | 0.56486 | **0.57711** | **+0.01225** |
| B-cubed F1 | 0.85259 | **0.86477** | **+0.01218** |
| False-merge pair rate | **0.08230** | 0.11597 | +0.03366 |
| False-split group rate | 0.27181 | **0.20081** | **-0.07099** |

Epoch 1 is the only checkpoint that passes every locked gate. Later epochs achieve higher recall
but lose excessive precision: false merge rises to `0.15129` at epoch 2 and `0.18461` at epoch 3.
Early stopping therefore retains epoch 1. The improvement exceeds the minimum recall gate by only
`0.00039`, so it is an accepted development candidate, not a replacement for the canonical system.
Independent confirmation remains required before any canonical change.

**Experiment closure:** this refinement cycle is closed as development-only evidence. The
full-joint checkpoint is not promoted, the canonical system remains unchanged, and the reserved
internal-confirmation partition has not been evaluated. Subsequent work uses the catalog-attachment
protocol as a separate evaluation track.

## Catalog-attachment protocol v4

Protocol v4 models a static catalog lookup rather than assuming every evaluation listing is both a
query and a candidate. The 1,542-listing development partition is assigned deterministically to
394 catalog references, 815 queries whose entity exists in the catalog, and 333 queries whose
entity is absent. The role manifest contains IDs and roles but no labels. TF-IDF remains train-fit,
and `label_group` enters only the protocol-construction and metric functions.

Every query retrieves exactly 20 unique reference candidates. The contract audit finds zero
query/reference ID overlap, self-candidates, duplicate candidates, candidates outside the reference
catalog, and train/development label overlap. Confirmation and historical test remain unaccessed.

| Development-v4 metric | Canonical | Full-joint candidate | Delta |
|---|---:|---:|---:|
| Retrieval Recall@1 | **0.78528** | 0.77669 | -0.00859 |
| Retrieval Recall@20 | 0.99018 | **0.99264** | +0.00245 |
| Attachment precision | 0.91766 | **0.93271** | +0.01504 |
| Attachment recall | **0.75215** | 0.73129 | -0.02086 |
| Attachment F1 | **0.82670** | 0.81981 | -0.00690 |
| New-entity detection recall | **0.87688** | 0.87087 | -0.00601 |
| New-entity false-attachment rate | 0.09610 | **0.06607** | -0.03003 |
| Overall false-attachment rate | 0.04791 | **0.03746** | -0.01045 |
| Manual-review rate | **0.05052** | 0.08188 | +0.03136 |

The canonical policy selects threshold `0.14`. The full-joint candidate selects `0.20` and passes
the absolute safety limits, but it fails the baseline-relative improvement gate: it becomes more
conservative instead of increasing safe attachment recall. The candidate is rejected for protocol
v4 and is not sent to confirmation. The canonical policy is the development winner; it is not yet
the final reported protocol until its policy and confirmation manifest are frozen.

No additional training is triggered at this stage because the canonical system already meets the
predeclared attachment, new-entity detection, false-attachment, and review-rate gates. Any later
training must target catalog attachment explicitly and use only development evidence.

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
.venv\Scripts\shopee-entity-resolution evaluate-hybrid-candidates `
  --config configs\experiment\entity_fragment_recovery_benchmark.yaml
.venv\Scripts\python -m shopee_match.training.hard_negative_cli train-hard-positive `
  --config configs\experiment\hard_positive_pair_finetuning.yaml
.venv\Scripts\python -m shopee_match.training.hard_negative_cli train-joint-recall `
  --config configs\experiment\full_joint_pair_recall_training.yaml
.venv\Scripts\python -m shopee_match.evaluation.catalog_attachment_cli build-protocol `
  --config configs\data\catalog_attachment_development_v4.yaml
.venv\Scripts\python -m shopee_match.evaluation.catalog_attachment_cli evaluate `
  --config configs\experiment\catalog_attachment_canonical_development.yaml
.venv\Scripts\python -m shopee_match.evaluation.catalog_attachment_cli evaluate `
  --config configs\experiment\catalog_attachment_full_joint_development.yaml
```

EfficientNet-B1 fine-tuning is deferred because the selected multi-source retrieval policy resolves
the measured validation bottleneck without reopening encoder training or adding that compute cost.
