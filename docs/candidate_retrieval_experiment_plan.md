# Candidate retrieval experiment plan

## Objective

Turn the canonical Phase 6 listing embedding into a reusable candidate-generation boundary. Measure
candidate Recall@K separately from pair classification, choose the smallest validation K that
reaches the configured recall target, and quantify the quality/latency/memory trade-off of FAISS
HNSW against a deterministic exact cosine reference.

## Frozen-source contract

- Load the SHA-256-locked canonical seed-2026 Phase 6 config, checkpoint, metrics, and mined-pair
  manifest.
- Extract one L2-normalized joint embedding per validation listing from the frozen modality cache.
- Never load or evaluate the held-out test split in Phase 7 model/index selection.
- Treat posting IDs as corpus metadata, not model inputs.

## Evaluation order

1. Build and serialize the exact cosine reference.
2. Verify exact search and exact-index round-trip on hand-checkable tests.
3. Sweep K on exact validation retrieval and choose the smallest K meeting target recall.
4. Only then build FAISS HNSW and sweep `efSearch`.
5. Select the least expensive HNSW setting that stays within the recall-drop and exact-candidate
   agreement gates.
6. Record batch throughput, single-query p50/p95 latency, embedding/index bytes, and failure strata.

## Exit criteria

- [x] Exact Top-K is deterministic and its serialized round-trip is identical.
- [x] FAISS and exact outputs agree on a small fixture.
- [x] Candidate Recall@K and the selected K are recorded on validation only.
- [x] Approximate recall and exact-candidate agreement pass configured gates.
- [x] Extraction/search throughput, p50/p95 latency, index size, and memory estimates are recorded.
- [x] Retrieval misses are categorized by group size, title length, and exact-pHash evidence.
- [x] Commands and results are documented before Phase 8 begins.

## Accepted validation result

- Exact search selects `K=50`: Recall@50 `0.97438`, hit rate `0.98863`, mAP@50 `0.87279`.
- HNSW selects `efSearch=64`: Recall@50 `0.97438`, exact candidate-set agreement `0.99851`.
- Exact/HNSW single-query p50 latency: `0.414 / 0.200 ms`; p95: `0.573 / 0.297 ms`.
- Exact/HNSW serialized size: `6,537,636 / 8,040,892` bytes.
- Exact retrieval has `39` zero-positive queries, `203` partial-group queries, and `3,188`
  complete-group queries at K=50.
- Held-out test status: `disabled_phase7_validation_only`; `test_accessed=false`.

Run manually after installing the retrieval extra:

```powershell
.venv\Scripts\python.exe -m pip install -e ".[dev,retrieval]"
.venv\Scripts\python.exe -m pytest -q --basetemp=.pytest_tmp\phase7 -p no:cacheprovider
.venv\Scripts\python.exe -m shopee_match.retrieval.cli benchmark --config configs\experiment\candidate_retrieval_benchmark.yaml
```

The benchmark refuses to overwrite completed metrics or the report. Change the output paths for an
intentional new run instead of silently replacing accepted evidence.
