# Phase 5 multimodal model final comparison

## Decision

Phase 5 is complete. The canonical model combines frozen custom image and text embeddings with a
score-preserving residual fusion module and a symmetric pair head. The checkpoint, configuration,
validation metrics, simple-fusion weight, and pair threshold were frozen before the held-out test
was evaluated once.

The model is valid and reproducible, but it does not beat the strongest classical fusion baseline.
That is an important result rather than a failed pipeline: learned fusion improves the custom
neural text/image combination, while the train-only character TF-IDF representation remains a very
strong signal for exact product identity.

## Validation ablations

| Configuration | Checkpoint target | mAP@20 | Recall@20 | Interpretation |
|---|---|---:|---:|---|
| Image only | retrieval | 0.53907 | 0.64667 | Visual evidence alone is insufficient. |
| Text only | retrieval | 0.75693 | 0.87414 | Stronger than the custom image encoder. |
| Simple score fusion | retrieval | 0.87358 | **0.94005** | No learned Phase 5 loss. |
| Contrastive only | learned fusion | 0.87358 | 0.94005 | Best checkpoint is initialization; training did not improve it. |
| Pair BCE only | pair-head rerank | 0.85092 | 0.93827 | Pair supervision alone damages ranking quality. |
| Combined losses, pair head off | learned fusion | 0.87023 | 0.93780 | Joint embedding trails simple fusion. |
| Combined losses, pair head on | pair-head rerank | **0.87903** | 0.93780 | Best learned validation ranking. |

The combined objective is necessary for the selected learned result. Supervised contrastive loss
preserves useful global structure, while the lower-weight pair BCE term improves local ordering.
The pair head raises canonical validation mAP@20 by `+0.00880` over its learned-fusion candidate
ordering and by `+0.00545` over simple score fusion. Recall@20 does not increase because reranking
cannot add a missing candidate.

## Repeated-seed stability

| Statistic over seeds 2026/2027/2028 | mAP@20 | Recall@20 |
|---|---:|---:|
| Mean | 0.88008 | 0.93838 |
| Sample standard deviation | 0.00132 | 0.00057 |

The original seed 2026 remains canonical; no seed was selected after comparing results. The small
standard deviations show that the validation result is not a lucky initialization.

## Frozen held-out test

| Method | Test mAP@20 | Test Recall@20 | Test pair F1 |
|---|---:|---:|---:|
| Custom image encoder | 0.55675 | 0.65941 | 0.4897 (Phase 3 threshold) |
| Custom text encoder | 0.74841 | 0.86978 | 0.5619 (Phase 4 threshold) |
| Phase 5 simple custom fusion | 0.86277 | 0.93098 | not selected |
| Phase 5 learned fusion | 0.85946 | 0.93235 | not selected |
| **Phase 5 pair-head rerank** | **0.86848** | **0.93235** | **0.68429** |
| Classical TF-IDF | 0.8564 | 0.9291 | 0.7048 |
| Classical pHash/ORB/TF-IDF fusion | **0.8810** | **0.9349** | **0.7220** |

The canonical Phase 5 model exceeds TF-IDF by approximately `+0.0121` test mAP@20 and `+0.0033`
Recall@20, but trails classical fused retrieval by approximately `-0.0125` mAP@20, `-0.0025`
Recall@20, and `-0.0377` pair F1. Validation-to-test mAP changes from `0.87903` to `0.86848`
(`-0.01054`), which is a modest generalization gap rather than evidence of leakage or fabricated
performance.

## Categorized validation failures

| Diagnostic | Queries | Share |
|---|---:|---:|
| Pair-head Top-1 false match | 332 | 9.68% |
| No true duplicate in pair-head Top-20 | 68 | 1.98% |
| Pair head regresses a correct simple-fusion Top-1 | 32 | 0.93% |
| Pair head rescues an incorrect simple-fusion Top-1 | 54 | 1.57% |
| Image correct while text is wrong at Top-1 | 366 | 10.67% |
| Text correct while image is wrong at Top-1 | 1,022 | 29.80% |
| Image/text Top-1 disagreement | 2,590 | 75.51% |
| False pair-head Top-1 with digit/unit conflict | 220 | 6.41% |

The next experiment should therefore target hard negatives involving model numbers, quantity,
volume, size, and units. Phase 6 should mine these from training only and measure whether precision
improves without sacrificing the current candidate recall.

## Metric hierarchy

For the Phase 5 retrieval model, frozen-test mAP@20 is the primary quality metric because it rewards
both early ordering and recovery of all relevant duplicates. Recall@20 is the candidate-ceiling
metric and must accompany mAP, but a high value alone is not proof of accurate ranking. Pair F1 at
the validation-frozen threshold is the main operating-point metric; precision and recall must be
reported beside it because the same F1 can hide different business risks. Repeated-seed mean and
standard deviation establish stability, while the untouched test protocol provides the strongest
current guard against optimistic validation selection.

No single scalar is universally objective. In later entity-resolution phases, pairwise/B-cubed F1
and false-merge rate become more important than retrieval mAP because one false-positive edge can
merge entire product groups.
