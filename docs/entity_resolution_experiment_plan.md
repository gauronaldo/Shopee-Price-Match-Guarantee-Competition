# Entity Resolution Experiment Plan

## Objective

Convert the frozen Phase 7 Top-50 candidate sets into conservative product entities while
measuring pair decisions separately from cluster quality. Phase 8 uses validation only and does
not retrain the image, text, fusion, or pair-head models.

## Frozen inputs

- accepted Phase 6 symmetric pair-head checkpoint;
- Phase 7 normalized validation embeddings;
- exact Top-50 cosine candidate protocol;
- SHA-256-verified Phase 7 configuration, metrics, and embedding cache;
- group-disjoint validation labels for graph-policy selection and analysis only.

Test remains disabled. Labels must never enter pair features, reciprocal-neighbour checks, graph
construction, component consistency, confidence, or manual-review rules.

## Pair scoring

Directed Top-50 candidates are deduplicated into undirected pairs. Each pair is scored once using
the symmetric pair head over `[z1*z2, |z1-z2|]`. Saved evidence includes cosine similarity,
pair probability, both directional ranks, and a title-derived variant-conflict flag.

## Conservative graph

An edge is eligible only when:

1. pair probability exceeds the trial threshold;
2. each endpoint occurs within the other's configured reciprocal rank;
3. a quantity/model-token conflict exceeds the stricter override probability.

Eligible edges are processed strongest first. Union-find proposes component merges. A merge is
blocked when it exceeds the maximum cluster size or when too few members on either side have a
supporting cross-component edge. This explicitly prevents a single bridge from joining two
established components.

## Selection

Validation sweeps pair thresholds, reciprocal ranks, and component-coverage requirements. The
selected policy maximizes B-cubed F1 subject to pairwise cluster precision of at least `0.90`.
Ties prefer pairwise F1, higher precision, lower split rate, higher threshold, smaller reciprocal
rank, and stronger component coverage.

## Metrics and exit gate

- accepted-edge precision, recall, and F1 using all true validation pairs as recall denominator;
- pairwise clustering precision, recall, and F1;
- B-cubed precision, recall, and F1;
- false-merge pair rate and impure-cluster rates;
- false-split group rate and group-size-stratified fragmentation;
- graph rejection counters, cluster sizes, confidence, and manual-review flags.

Phase 8 passes when the selected validation graph reaches the configured pairwise precision gate,
all metrics and failures are persisted, labels are demonstrably analysis-only, tests pass, and
test remains untouched.

## Manual command

```powershell
.venv\Scripts\shopee-entity-resolution benchmark --config configs\experiment\entity_resolution_benchmark.yaml
```
