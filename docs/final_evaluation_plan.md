# Final Evaluation and Portfolio Packaging Plan

## Objective

Freeze the accepted custom system and evaluate its complete retrieval, pair-scoring, and
entity-resolution path on the held-out split exactly once. The test result is descriptive: it may
not change candidate K, pair threshold, graph rules, model weights, or any later claim.

## Frozen source

The final system reuses the clean accepted entity-resolution validation run and recursively locked
sources from candidate retrieval and hard-negative pair-head training. The explicit final config
must exactly reproduce these validation-selected values:

- exact cosine candidate budget: Top-50;
- pair probability threshold: `0.16`;
- reciprocal-neighbour rank: `5`;
- full cross-component coverage: `1.0`;
- variant-conflict threshold: `0.15`;
- maximum cluster size: `64`;
- manual-review margin: `0.02`.

The entity config and metrics are SHA-256 verified. The source validation metrics must report a
clean Git worktree, a passed pairwise-precision gate, and no test access.

## One-time access guard

`preflight` verifies hashes, the frozen policy, output absence, Git cleanliness, and CUDA without
loading test rows. `evaluate` refuses a dirty worktree and writes an ignored access marker before
loading the held-out split. If the process fails after that marker, automatic rerun is blocked and
the incident must be documented rather than silently deleting the marker.

Earlier phases already evaluated image, text, and multimodal components on this same split. The
final run is therefore the first complete system evaluation, but the split is not globally unseen
to the project owner. This limitation must remain explicit in the final report.

## Metrics

- retrieval: mAP@20/50, Recall@1/5/10/20/50, Precision and F1 at K;
- candidate-conditioned pair head: precision, recall, F1, average precision/PR-AUC, Brier score,
  expected calibration error, and controlled operating points;
- accepted graph edges: global precision, recall, and F1 with all true pairs as denominator;
- entities: pairwise and B-cubed precision/recall/F1, false-merge rate, false-split rate, cluster
  sizes, manual-review counts, and group-size strata;
- efficiency: encoder/fusion/pair throughput, exact-query latency, embedding size, parameters, and
  end-to-end wall time.

## Pre-test gates

- [x] full unit/integration suite, Ruff, and mypy pass;
- [x] hand-computed pair classification and clustering fixtures pass;
- [x] config loader rejects any policy drift or non-clean validation source;
- [x] CLI wrapper is installed; pre-test preflight reported no prior access and loaded no test rows;
- [x] implementation and frozen config are committed before test access;
- [x] worktree is clean and no final output/access marker exists.

## Post-test packaging

- [x] record the immutable aggregate final report;
- [x] consolidate final model card, limitations, ethical considerations, and dataset access terms;
- [x] summarize repeated-seed and ablation evidence already produced in earlier phases;
- [x] add architecture and inference-flow diagrams;
- [ ] verify commands from a clean clone after the user supplies the competition dataset;
- [x] audit Git for secrets, datasets, checkpoints, caches, oversized outputs, and unintended
  provenance metadata before merging.

## Manual commands

```powershell
.venv\Scripts\python -m pip install -e ".[dev,retrieval,pretrained]"
.venv\Scripts\ruff check .
.venv\Scripts\mypy src
.venv\Scripts\python -m pytest
.venv\Scripts\shopee-final preflight `
  --config configs\experiment\final_system_evaluation.yaml

# Run only after the frozen implementation/config commit and a ready preflight.
.venv\Scripts\shopee-final evaluate `
  --config configs\experiment\final_system_evaluation.yaml
```
