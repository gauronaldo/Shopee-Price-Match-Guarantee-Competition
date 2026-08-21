# Hard-negative mining

## Outcome

Phase 6 mines deceptively similar different-product pairs from train only, then fine-tunes the
symmetric pair head while keeping the fusion embedding frozen. The accepted method passes all
three deterministic seeds: controlled-recall precision improves modestly, mAP@20 does not regress,
variant-conflict errors decrease, and Recall@20 stays fixed.

## Mining contract and result

The frozen Phase 5 model retrieves 100 exact neighbours for every train listing. Labels are used
only after retrieval to keep cross-label pairs. Same-pHash and exactly normalized cross-label
titles are excluded as possible false negatives or fragmented labels; symmetric duplicates are
collapsed.

| Measure | Value |
|---|---:|
| Train listings queried | 27,391 |
| Raw retrieved candidates | 2,739,100 |
| Eligible cross-label candidates | 766,561 |
| Final unique mined pairs | 24,332 |
| Digit/unit variant-conflict pairs | 12,166 (50%) |
| Symmetric duplicates removed | 13,828 |
| Pair probability median / P95 | 0.33214 / 0.44949 |
| Mining wall time | approximately 38-40 s |

| False-negative guard | Excluded pairs |
|---|---:|
| Same label | 130,837 |
| Outside probability bounds | 1,840,596 |
| Same pHash across labels | 690 |
| Exact normalized title across labels | 416 |
| Variant candidates removed by 50% share cap | 29,144 |

Test data is not loaded during mining or Phase 6 selection.

## Pilot decision

The first pilot updated both fusion and pair-head weights. Although its retained initialization did
not regress mAP or Recall, it produced no controlled-precision or variant improvement; the pilot
failed. Later epochs reduced validation mAP from `0.87396` to `0.86408`, showing that joint updates
distorted the retrieval representation.

The accepted method therefore freezes fusion and updates only the pair head using a 75/25 mix of
original/random-pair BCE and mined hard-negative BCE. Supervised-contrastive loss remains a
diagnostic rather than an optimized term.

## Repeated-seed validation

Frozen Phase 5 reference: mAP@20 `0.87903`, controlled-recall precision `0.74557`, Recall@20
`0.93780`, and 280 false Top-1 variant conflicts.

| Seed | Best epoch | mAP@20 | mAP delta | Controlled precision | Precision delta | Variant errors | Variant delta | Gate |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 2026 | 1 | 0.87935 | +0.00032 | 0.74847 | +0.00290 | 276 | -4 | pass |
| 2027 | 1 | 0.87917 | +0.00014 | 0.74785 | +0.00228 | 278 | -2 | pass |
| 2028 | 1 | 0.87925 | +0.00023 | 0.74772 | +0.00215 | 277 | -3 | pass |

| Aggregate | Mean | Population standard deviation |
|---|---:|---:|
| mAP@20 | 0.87926 | 0.00007 |
| Controlled-precision delta | +0.00244 | 0.00033 |
| Variant Top-1 delta | -3.00 | not applicable |

Recall@20 is `0.93780` for the reference and every selected Phase 6 run because candidate
embeddings are frozen. All three runs select epoch 1 and then lose validation mAP while training
loss continues to fall, supporting early stopping.

## Canonical seed and provenance

Seed 2026 remains canonical because it was pre-declared, not because it has the largest score.

- Phase 5 source checkpoint SHA-256: `95289d84fbb85f99764f42b05ded92ec2c535b2b421b3fa1422cfb987b2800f4`
- Phase 5 source config SHA-256: `279c96794c207fb2e62e4638cdae315dc4ffd4a2b85ecf039f41861e7412377c`
- Phase 6 config SHA-256: `12b8b7e97815b8cd710f3a3a6160ca3263d36453face15627b2ecc17ccab94fa`
- Phase 6 checkpoint SHA-256: `d763834919c9bea2378b112e870d15b82817023692940c20f112f98d49370c3e`
- Phase 6 metrics SHA-256: `7bff1b804b0a611e1a5380540e54f3b6955da3c4b4ad42ff5d7aff2442e898ee`
- Mined manifest SHA-256: `ad716c1c7a4d5e1aa31cbd668c98b3d1c6f42117d865d2bc2aa5bf995e19d2d2`

Hard negatives cannot create missing candidates. They teach the pair classifier to demote
look-alike non-matches at a controlled recall level. The improvement is consistent but small and
must not be presented as a large model gain. Phase 6 is closed on validation; test remains
untouched.
