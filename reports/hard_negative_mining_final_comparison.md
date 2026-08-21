# Hard-negative mining final comparison

## Outcome

Phase 6 **passes on validation across all three deterministic seeds**. Pair-head-only fine-tuning
improved precision at the frozen Phase 5 recall operating point in every run, preserved candidate
Recall@20 exactly, and did not regress mAP@20. The effect is deliberately reported as modest.

The earlier joint fusion/pair-head pilot failed because it distorted the retrieval embedding; its
failure is retained in `hard_negative_mining_pilot.md`. The accepted method freezes fusion and
updates only the symmetric pair classifier using a 75/25 mix of original/random-pair and mined
hard-negative BCE.

## Frozen Phase 5 reference

| Metric | Value |
|---|---:|
| Pair-head mAP@20 | 0.87903 |
| Controlled-recall precision | 0.74557 |
| Pair-head Recall@20 | 0.93780 |

## Repeated-seed validation

| Seed | Best epoch | mAP@20 | mAP delta | Controlled precision delta | Recall@20 delta | Variant Top-1 delta | Gate |
|---:|---:|---:|---:|---:|---:|---:|---|
| 2026 | 1 | 0.87935 | +0.00032 | +0.00290 | +0.00000 | -4 | pass |
| 2027 | 1 | 0.87917 | +0.00014 | +0.00228 | +0.00000 | -2 | pass |
| 2028 | 1 | 0.87925 | +0.00023 | +0.00215 | +0.00000 | -3 | pass |

| Aggregate | Mean | Population std. |
|---|---:|---:|
| mAP@20 | 0.87926 | 0.00007 |
| Controlled precision delta | +0.00244 | 0.00033 |
| Variant Top-1 error delta | -3.00 | not applicable |

## Canonical Phase 6 artifact

Seed `2026` remains canonical because it is the project's pre-declared primary seed,
not because it produced the largest score.

- Config SHA-256: `12b8b7e97815b8cd710f3a3a6160ca3263d36453face15627b2ecc17ccab94fa`
- Checkpoint SHA-256: `d763834919c9bea2378b112e870d15b82817023692940c20f112f98d49370c3e`
- Metrics SHA-256: `7bff1b804b0a611e1a5380540e54f3b6955da3c4b4ad42ff5d7aff2442e898ee`
- Mined manifest SHA-256: `ad716c1c7a4d5e1aa31cbd668c98b3d1c6f42117d865d2bc2aa5bf995e19d2d2`
- Mined pairs: `24,332` (`50%` digit/unit variant conflicts)

## Scope and interpretation

Hard negatives do not create new candidate recall. They teach the pair classifier to demote
different products that the Phase 5 embedding retrieves as deceptively similar. Freezing fusion is
why Recall@20 stays fixed; only ordering and decision precision change. Validation alone selected
all checkpoints. Phase 6 did not access or retune on test.

The gain is statistically consistent across these deterministic sampling seeds but small; it should
not be described as a large model improvement. Phase 6 is closed, and Phase 7 may use the canonical
seed-2026 checkpoint for candidate-generation work.
