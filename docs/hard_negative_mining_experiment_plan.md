# Hard-negative mining experiment plan

## Objective

Improve pair precision on visually or lexically similar non-matches without sacrificing the
candidate recall established in Phase 5. Mining and training use only the train split. Validation
selects the checkpoint and measures the trade-off; the held-out test remains closed.

## Mining contract

- Load the SHA-256-locked canonical Phase 5 checkpoint.
- Extract its joint train embeddings from the existing frozen-encoder cache.
- Retrieve exact nearest neighbours in bounded matrix blocks.
- Keep only cross-label candidates with high pair-head probability.
- Exclude identical pHash and identical normalized-title cross-label pairs as possible label noise.
- Exclude extremely high-probability cross-label pairs from training and retain them for review.
- Prefer a bounded share of candidates with conflicting digits or units.
- Deduplicate symmetric pairs and write a deterministic JSONL manifest plus provenance metadata.

## Training contract

- Initialize from the canonical Phase 5 checkpoint, not random weights.
- Preserve product-aware batches and random in-batch negatives. Measure supervised contrastive
  loss as a diagnostic when fusion is frozen.
- Add a configured number of mined negative pairs to every optimization step.
- Mix random-negative and hard-negative BCE rather than replacing one with the other.
- Select checkpoints on validation pair-head mAP@20; never access test.

## Success criteria

- Improve precision at the Phase 5 validation recall operating point.
- Do not reduce validation mAP@20 below the Phase 5 reference `0.87903`.
- Keep Recall@20 within `0.002` of the Phase 5 reference `0.93780`.
- Reduce variant-conflict and pair-head-regression failures in categorized validation analysis.
- If the pilot passes, verify the improvement across repeated seeds before closing Phase 6.

## Gates

- [x] Strict config and source hashes validate.
- [x] Miner is deterministic and train-only.
- [x] False-negative guards and symmetric deduplication are tested.
- [x] Mixed random/hard pair loss has gradient-flow and determinism tests.
- [x] Real-data mining report is recorded.
- [x] Validation-only pilot is compared with the frozen Phase 5 baseline.
- [x] Failure analysis explains whether Phase 6 passes or fails.

## Recorded outcome

The first pilot updated both fusion and pair head. It failed: validation mAP@20 declined from
`0.87903` to `0.87396` after one epoch and continued downward, so early stopping retained the
unchanged Phase 5 initialization. This showed that the mined-pair signal was distorting candidate
retrieval.

The accepted experiment freezes fusion and trains only the symmetric pair head with `75%`
original/random-pair BCE and `25%` mined hard-negative BCE. Seeds `2026`, `2027`, and `2028` all
pass. Mean mAP@20 is `0.87926 +/- 0.00007`; controlled-recall precision improves by
`+0.00244 +/- 0.00033`; Recall@20 is unchanged; and false Top-1 variant conflicts fall by `2-4`
queries per seed. Test was not accessed. Phase 6 is closed on this validation-only evidence.
