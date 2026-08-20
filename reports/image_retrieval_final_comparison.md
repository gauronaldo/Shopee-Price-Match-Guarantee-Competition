# Final Phase 3 image-retrieval comparison

## Outcome

The scratch residual image encoder was trained from random initialization with no title features,
pretrained weights, pHash, or ORB scores. Its SHA-256-locked checkpoint reached validation/test
mAP@20 of `0.53907 / 0.55674` and Recall@20 of `0.64667 / 0.65941`. Test quality is slightly higher
than validation, so the frozen checkpoint shows no evidence of validation overfitting.

## Retrieval comparison

| Method | Evidence used by end-to-end pipeline | Validation mAP@20 | Validation Recall@20 | Test mAP@20 | Test Recall@20 |
|---|---|---:|---:|---:|---:|
| Supplied pHash | Image hash | 0.2895 | 0.3174 | 0.3073 | 0.3345 |
| Scratch residual CNN | Image embedding | **0.5391** | **0.6467** | **0.5567** | **0.6594** |
| ORB pipeline | ORB reranking over pHash + TF-IDF candidate union | 0.6638 | 0.8284 | 0.6577 | 0.8151 |
| Character TF-IDF | Listing title | 0.8635 | 0.9385 | 0.8564 | 0.9291 |
| Late fusion | Image + title | 0.8790 | 0.9411 | 0.8810 | 0.9349 |

On test, the scratch encoder improves over pHash by `+0.2494` mAP and `+0.3249` Recall@20. It
trails the ORB pipeline by `0.1010` mAP and `0.1557` Recall@20. The ORB comparison is not a pure
image-only contest: its candidate union includes TF-IDF title retrieval, whereas the scratch model
performs exact image-embedding search over the full split. This protocol caveat must accompany any
claim that ORB is stronger.

TF-IDF and fusion are system-level context rather than Phase 3 competitors. Their advantage shows
that exact-product identity in this catalog depends heavily on title evidence such as brand,
quantity, unit, and model tokens.

## Frozen-threshold pair comparison

| Method | Test pair precision | Test pair recall | Test pair F1 |
|---|---:|---:|---:|
| pHash | — | — | 0.3607 |
| Scratch residual CNN | **0.8323** | 0.3469 | **0.4897** |
| ORB pipeline | — | — | 0.5766 |
| Character TF-IDF | — | — | 0.7048 |
| Late fusion | — | — | 0.7220 |

The scratch threshold `0.805664` was selected on validation and applied unchanged to test. Test
precision increased from validation `0.7881` to `0.8323`, while recall fell from `0.3810` to
`0.3469`; pair F1 moved from `0.5137` to `0.4897`. The image-only score is therefore useful as a
high-confidence signal but too conservative to serve as the final entity-resolution decision by
itself.

## Generalization and strata

| Stratum | Validation mAP@20 | Test mAP@20 |
|---|---:|---:|
| Group size 2 | 0.4760 | 0.5064 |
| Group size 3–5 | 0.5845 | 0.5975 |
| Group size 6–9 | 0.6225 | 0.6091 |
| Group size 10+ | 0.5369 | 0.5583 |
| Has exact-pHash positive | 0.8333 | 0.8304 |
| No exact-pHash positive | 0.4389 | 0.4681 |

Performance is stable across validation and test. Groups of two remain the weakest mAP band, and
the large exact-positive/no-exact-positive gap confirms continued dependence on near-duplicate
visual evidence. Nevertheless, test no-exact-positive mAP `0.4681` demonstrates substantial
generalization beyond direct image duplication.

## Efficiency

| Measurement | Validation | Test |
|---|---:|---:|
| Embedding throughput | 270.50 listings/s | 159.93 listings/s |
| Exact ranking p50 | 0.442 ms/query | 0.450 ms/query |
| Exact ranking p95 | 0.656 ms/query | 0.717 ms/query |
| Embedding storage | 3.35 MiB | 3.35 MiB |

The model has `3.06M` parameters, a `35.17 MiB` checkpoint, and used approximately `1.01 GiB`
peak allocated CUDA memory during training. Ranking is fast once embeddings exist. Extraction
throughput varies with loader/runtime conditions and should not be presented as a hardware-neutral
benchmark.

## Reproducibility and test policy

- Frozen checkpoint SHA-256: `6ea26b493d643b148cbcc48006231637b266491a0a026d7fdbd22284f7100e07`.
- Frozen training config SHA-256: `7d9551060b2b47a023eaf00d39f92fbb2174102914101a613cfa2f8a1cb8c06a`.
- Frozen training metrics SHA-256: `3b389c5dd6cb58548249931fa77b1aa6b5821d540f07f9d99294e8725cee2a6a`.
- Test protocol: exact full-split Top-20 retrieval, self excluded.
- Test threshold source: validation only; no test-time selection.
- Test was evaluated once after the three hashes and threshold were recorded.

## Phase 3 decision

The scratch image encoder provides a clear measured advance over pHash, stable held-out
generalization, compact embeddings, and a documented quality gap to the candidate-assisted ORB
pipeline. The failure taxonomy explains that global-layout shortcuts and coarse-category matches
remain dominant. Phase 3 is closed; no additional test tuning or rerun is permitted. The next
modeling step is an independently evaluated scratch text encoder.
