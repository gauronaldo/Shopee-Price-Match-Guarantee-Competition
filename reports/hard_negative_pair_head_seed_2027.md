# Hard-negative mining pilot

## Contract

The frozen Phase 5 model retrieved neighbours from the **train split only**. Labels were used only
after retrieval to retain different-product pairs. Test data was not loaded. Cross-label pairs with
the same pHash or exactly normalized title were excluded as possible false negatives or fragmented
labels. Symmetric duplicates were collapsed into one canonical pair.

## Mining result

| Measure | Value |
|---|---:|
| Train listings queried | 27,391 |
| Exact neighbours per query | 100 |
| Raw retrieved candidates | 2,739,100 |
| Eligible cross-label candidates | 766,561 |
| Final unique mined pairs | 24,332 |
| Variant-conflict pairs | 12,166 |
| Symmetric duplicates removed | 13,828 |
| Pair probability median / P95 | 0.33214 / 0.44949 |
| Mining wall time | 38.73 s |

## False-negative guards

| Exclusion | Count |
|---|---:|
| Same label (true train positive) | 130,837 |
| Outside probability bounds | 1,840,596 |
| Same pHash across labels | 690 |
| Exact normalized title across labels | 416 |
| Variant candidates removed by final share cap | 29,144 |

## Provenance

- Source checkpoint SHA-256: `95289d84fbb85f99764f42b05ded92ec2c535b2b421b3fa1422cfb987b2800f4`
- Source config SHA-256 (canonical LF): `279c96794c207fb2e62e4638cdae315dc4ffd4a2b85ecf039f41861e7412377c`
- Mined manifest SHA-256: `ad716c1c7a4d5e1aa31cbd668c98b3d1c6f42117d865d2bc2aa5bf995e19d2d2`
- Split: `train`
- Test accessed: `false`

The section above records mining evidence. The Phase 6 training command appends its validation
comparison below without changing the mined manifest.

## Fine-tuning result

The canonical Phase 5 weights initialized this run. Every optimization step retained the original
product-aware supervised-contrastive batch and random in-batch pair examples, then added a separate
batch of mined non-matches. The hard-negative BCE contributes only the configured share of the pair
loss; it does not replace the original training signal.

The fusion embedding was frozen after the joint pilot regressed. Only the pair head was optimized; supervised-contrastive loss was still measured as a diagnostic.

| Validation measure | Frozen Phase 5 | Phase 6 selected | Delta |
|---|---:|---:|---:|
| Pair-head mAP@20 | 0.87903 | 0.87917 | +0.00014 |
| Pair-head Recall@20 | 0.93780 | 0.93780 | +0.00000 |
| Precision at controlled recall | 0.74557 | 0.74785 | +0.00228 |
| False Top-1 variant conflicts | 280 | 278 | -2 |

### Acceptance gates

- mAP non-regression: `true`
- Recall@20 drop no greater than 0.002: `true`
- Precision improved at the Phase 5 recall target: `true`
- Variant-conflict errors did not increase: `true`
- Pilot outcome: **pilot_pass_requires_repeated_seed_confirmation**

## Training history

| Epoch | Total loss | Random-pair BCE | Hard-negative BCE | Validation mAP@20 |
|---:|---:|---:|---:|---:|
| 1 | 0.30646 | 0.29458 | 0.34212 | 0.87917 |
| 2 | 0.29663 | 0.28992 | 0.31677 | 0.87914 |
| 3 | 0.28966 | 0.27750 | 0.32614 | 0.87789 |
| 4 | 0.28230 | 0.26496 | 0.33433 | 0.87654 |
| 5 | 0.27979 | 0.25814 | 0.34474 | 0.87533 |

## Interpretation

`mAP@20` remains the checkpoint-selection metric because it measures ranking quality over all
queries. The controlled-recall precision is the Phase 6 diagnostic: it asks whether the system can
reject more look-alike non-matches while preserving the Phase 5 match-recall operating point. A
pilot is not considered closed evidence until its improvement is repeated across seeds.
