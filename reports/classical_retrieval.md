# Classical retrieval benchmark

This report contains aggregate validation/test results only. Retrieval uses the full
corresponding split as its candidate pool and always excludes the query itself.
TF-IDF vocabulary and IDF are fit on train only. Fusion weight and pair
thresholds are selected on validation, then frozen for the final test evaluation.

## Evaluation protocol

- Configuration: `configs/experiment/classical_retrieval_benchmark.yaml`
- Seed: `2026`
- Environment: Python `3.12.13`, OpenCV `4.12.0`, NumPy `2.2.6`

## Results

Metrics are macro-averaged per query. Pair F1 counts unretrieved positives as false negatives. Retrieval columns use K=20.

| Baseline | Val mAP | Val recall | Val threshold | Test mAP | Test recall | Test pair F1 | End-to-end runtime (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| phash | 0.2895 | 0.3174 | 0.8125 | 0.3073 | 0.3345 | 0.3607 | 14.40 |
| tfidf | 0.8635 | 0.9385 | 0.4269 | 0.8564 | 0.9291 | 0.7048 | 89.66 |
| orb | 0.6638 | 0.8284 | 0.0280 | 0.6577 | 0.8151 | 0.5766 | 553.08 |
| fusion | 0.8790 | 0.9411 | 0.3687 | 0.8810 | 0.9349 | 0.7220 | 109.32 |

Selected fusion text weight: **0.75**.
Fusion improves test mAP@20 over TF-IDF by **0.0246**.
Runtime covers validation plus test; ORB and fusion include their candidate stages.
Mean end-to-end milliseconds/query: phash=2.10, tfidf=13.07, orb=80.64, fusion=15.94.
Peak process working set: **518.1 MiB**.

![Validation threshold sweeps](figures/classical_retrieval_threshold_sweeps.svg)

## Sampled failure analysis

Manual review found semantically unrelated pHash neighbors, title matches that omit
identity-critical model/variant details, and ORB matches driven by shared visual structure.
Several high-scoring cross-label title pairs also look plausibly identical, consistent with the
data-audit label-fragmentation warning. These cases remain evaluation errors; labels are not
silently rewritten.

## Interpretation

- The supplied pHash is an image-appearance signal, not proof of product identity.
- ORB reranks the label-blind union of pHash and TF-IDF candidates; its
  retrieval ceiling is therefore limited by that candidate union.
- Test labels were used only after validation selected the fusion weight and
  thresholds.
