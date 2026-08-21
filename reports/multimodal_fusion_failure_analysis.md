# Multimodal fusion validation failure analysis

## Scope

This analysis uses only the validation split and the selected Phase 5 checkpoint. Categories can
overlap because one query may exhibit both a modality disagreement and a variant-token conflict.
The title-rich review records are local artifacts and are not included in Git.

## Retrieval context

| Method | mAP@20 | Recall@20 |
|---|---:|---:|
| Image only | 0.53907 | 0.64667 |
| Text only | 0.75693 | 0.87414 |
| Simple score fusion | 0.87358 | 0.94005 |
| Learned fusion | 0.87023 | 0.93780 |
| Pair-head rerank | 0.87903 | 0.93780 |

## Categorized diagnostics

| Category | Queries | Share of validation queries |
|---|---:|---:|
| Pair Top1 False Match | 332 | 9.68% |
| Pair Retrieval Miss | 68 | 1.98% |
| Pair Head Regression | 32 | 0.93% |
| Pair Head Rescue | 54 | 1.57% |
| Image Rescue | 366 | 10.67% |
| Text Rescue | 1,022 | 29.80% |
| Modality Disagreement | 2,590 | 75.51% |
| Variant Token Conflict | 220 | 6.41% |

## Interpretation

- `pair top1 false match` is a ranking error at the first result, not necessarily a complete
  retrieval failure.
- `pair retrieval miss` means no true duplicate appears in the pair head's Top-20 candidate list.
- `pair head regression/rescue` measures whether learned reranking hurts or fixes the simple-fusion
  Top-1 result.
- `image rescue` and `text rescue` expose cases where one modality is correct and the other is not.
- `variant token conflict` is an automatic diagnostic: the false Top-1 pair differs in digits or
  units. It is evidence for manual review, not a semantic ground-truth category.

The key Phase 5 risk is the pair head improving average ranking while moving a subset of already
correct simple-fusion queries in the wrong direction. Phase 6 hard-negative mining should target
these regressions, especially variant conflicts involving model numbers, quantity, size, or unit.
