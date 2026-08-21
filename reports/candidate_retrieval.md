# Candidate retrieval benchmark

## Outcome

Phase 7 status: **phase7_complete_validation_only**. The benchmark uses only validation to select candidate K and FAISS
`efSearch`; held-out test data is not evaluated. Exact cosine search is the quality reference, and
FAISS HNSW is accepted only if it preserves recall and candidate-set agreement.

## Exact candidate ceiling

| K | Recall@K | Hit rate@K | mAP@K |
|---:|---:|---:|---:|
| 1 | 0.48247 | 0.89388 | 0.89388 |
| 5 | 0.80275 | 0.95481 | 0.86673 |
| 10 | 0.88278 | 0.96968 | 0.86512 |
| 20 | 0.93780 | 0.98017 | 0.87023 |
| 50 | 0.97438 | 0.98863 | 0.87279 |
| 100 | 0.98424 | 0.99125 | 0.87446 |

- Target macro candidate recall: `0.950`
- Selected exact K: `50`
- Target reached: `true`

## FAISS HNSW sweep at selected K

| efSearch | Recall | Delta vs exact | Exact candidate agreement | Gate |
|---:|---:|---:|---:|---|
| 16 | 0.97013 | -0.00424 | 0.90268 | false |
| 32 | 0.97452 | +0.00014 | 0.98100 | false |
| 64 | 0.97438 | +0.00000 | 0.99851 | true |
| 128 | 0.97438 | +0.00000 | 0.99989 | true |

- Selected `efSearch`: `64`
- Index: HNSW Flat, inner product over L2-normalized embeddings
- HNSW M / efConstruction / rerank buffer: `32` / `200` / `32`

## Efficiency

| Measure | Exact | FAISS HNSW |
|---|---:|---:|
| Single-query p50 latency | 0.414 ms | 0.200 ms |
| Single-query p95 latency | 0.573 ms | 0.297 ms |
| Batch throughput | 4067.76 queries/s | 7024.92 queries/s |
| Estimated in-memory index | 7,078,617 bytes | 8,009,859 bytes |
| Serialized index | 6,537,636 bytes | 8,040,892 bytes |

- Embedding extraction: `16467.92` listings/s
- Embedding matrix: `7,024,640` bytes
- Embedding dimension: `512`

## Retrieval failures at exact K=50

| Category | Queries |
|---|---:|
| No positive candidate retrieved | 39 |
| Some but not all group members retrieved | 203 |
| Complete group retrieved | 3188 |

The local failure-review JSON contains bounded title-rich examples and approximate disagreements;
it remains ignored because it is generated evidence. Aggregate group-size, title-length, and
exact-positive-pHash strata are stored in metrics.

## Interpretation

Recall@K is the Phase 7 primary metric because a match omitted here cannot be recovered by the pair
classifier in Phase 8. Hit rate is less strict: it needs only one duplicate, while macro Recall@K
rewards retrieving the full product group. HNSW changes candidate generation only; it does not apply
the Phase 6 pair head or make match/no-match decisions.

## Limitations

- These are validation-only model/index-selection results, not a new held-out test result.
- The measured catalog has `3,430` listings; latency and memory must be
  remeasured before claiming behavior at production catalog scale.
- Timings are hardware- and environment-specific. Quality gates, serialized round-trip checks, and
  exact agreement are the portable parts of this benchmark.
