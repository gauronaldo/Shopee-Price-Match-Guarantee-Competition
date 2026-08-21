# Reports index

The showcase keeps one reviewed Markdown report per system function. Pilot runs, repeated seeds,
ablations, final evaluation, efficiency, and failure analysis are combined inside that function's
report instead of being published as separate files.

| Function | Consolidated report | Included evidence |
|---|---|---|
| Data quality and leakage-safe split | [Data quality and split](data_quality_and_split.md) | Audit, split statistics, warnings, and figures |
| Classical retrieval | [Classical retrieval](classical_retrieval.md) | pHash, ORB, TF-IDF, late fusion, efficiency, and failures |
| Custom image encoder | [Image encoder](image_encoder.md) | Design, smoke/pilot/full training, frozen test, comparison, and failures |
| Custom text encoder | [Text encoder](text_encoder.md) | Design, smoke/pilot/full training, TF-IDF comparison, frozen test, and failures |
| Multimodal model | [Multimodal model](multimodal_model.md) | Fusion pilots, loss ablations, seeds 2026-2028, frozen test, and failures |
| Hard-negative mining | [Hard-negative mining](hard_negative_mining.md) | Mining audit, rejected pilot, accepted pair-head method, and seeds 2026-2028 |
| Candidate generation | [Candidate retrieval](candidate_retrieval.md) | Exact/FAISS selection, efficiency, agreement, and retrieval failures |
| Entity resolution | [Entity resolution](entity_resolution.md) | Pair scoring, reciprocal graph selection, clustering metrics, and failures |
| Pretrained comparison | [Pretrained benchmark](pretrained_benchmark.md) | Frozen EfficientNet-B1 quality, efficiency, domain gap, and scratch comparison |

## Evidence policy

- These consolidated reports contain the lightweight evidence intended for version control.
- Full epoch histories, checkpoints, indexes, embeddings, detailed metrics, and sampled review
  records remain under the ignored `artifacts/` tree.
- Frozen experiment configs retain historical report-output paths because those paths contribute
  to their SHA-256 provenance. If an older command regenerates a superseded root-level Markdown
  file, `.gitignore` keeps it local.
- Figures and aggregate tables remain in `figures/` and `tables/`.

This layout reduces dozens of overlapping Markdown files to nine functional reports without
combining unrelated experiments or hiding weak results.
