# End-to-End Demonstration

## Purpose and boundary

The demo turns the frozen research pipeline into an inspectable application without retraining,
retuning thresholds, or reopening the held-out test evaluation. It uses the 3,430-listing
validation catalog because Phase 11 is presentation work, not another model-selection step.

The user provides an image, a title, or both. Image-only and text-only requests return Top-K
similarity candidates. A request containing both modalities additionally returns pair
probabilities, image/title/joint similarities, a predicted catalog entity or no-match state, and a
manual-review flag. A label-blind catalog adapter retains only `posting_id`, `image`,
`image_phash`, and `title`; `label_group` is not retained by the serving runtime.

## Runtime stages

1. Available inputs are encoded by the custom residual CNN, character TextCNN, or both.
2. A unimodal request retrieves candidates directly from the corresponding normalized catalog
   embeddings and stops before pair or entity decisions.
3. For a multimodal request, the accepted residual fusion module creates one 512-dimensional
   listing embedding.
4. FAISS HNSW retrieves 50 candidates from the frozen validation catalog.
5. The symmetric pair head scores each query-candidate pair.
6. The frozen probability threshold and reciprocal Top-5 rule decide accepted matches.
7. Accepted candidates attach the query to an existing conservative catalog entity. Conflicting
   entity evidence, boundary scores, and flagged catalog clusters trigger manual review.

The displayed `match_probability` is a model score at a frozen operating point. It is not a legal,
commercial, or statistical guarantee that two listings are identical.

## Guided and open-ended use

The Streamlit client has two tabs:

- `Guided demo` exposes six curated validation scenarios covering near duplicates, varied photos,
  noisy wording, and variant/model confusion. Every scenario offers three catalog-backed query
  choices representing three different products, not three listings from one duplicate group.
  Only the sample `posting_id` and scenario text are versioned; images and product titles remain in
  the local competition data.
- `Upload your own` accepts an image, title, or both without assuming dataset knowledge.

Guided requests include their catalog `posting_id`, and every retrieval backend excludes that ID
before ranking. This prevents a trivial similarity-1.0 self-match. Each result then places the
query and candidate image/title side by side with the measured evidence below them.
Literal UTF-8 byte escapes in source titles are decoded for display only; the text encoder still
receives the original frozen representation.

## Local commands

Install the optional runtime dependencies:

```powershell
.venv\Scripts\python -m pip install -e ".[dev,retrieval,demo]"
```

Start both local services with one managed command:

```powershell
.venv\Scripts\python -m shopee_match.serving.cli launch
```

The launcher waits for API health before starting Streamlit. Press `Ctrl+C` once to terminate both
children. Use the separate commands below only when debugging one service.

Verify every source hash, load all models, and build the in-memory FAISS index:

```powershell
.venv\Scripts\python -m shopee_match.serving.cli preflight `
  --config configs\serving\demo.yaml
```

Start the API:

```powershell
.venv\Scripts\python -m shopee_match.serving.cli api `
  --config configs\serving\demo.yaml `
  --host 127.0.0.1 `
  --port 8000
```

Start the UI in a second terminal:

```powershell
$env:SHOPEE_DEMO_API_URL = "http://localhost:8000"
.venv\Scripts\streamlit run app\streamlit_app.py
```

Useful URLs:

- UI: `http://localhost:8501`
- API health: `http://localhost:8000/health`
- Interactive OpenAPI: `http://localhost:8000/docs`

Container alternative:

```powershell
docker compose config
docker compose up --build
```

Both containers use the same image. Compose mounts `data/` and `artifacts/` read-only because raw
competition files and checkpoints must never be baked into or committed with the application.
Docker Desktop must be running with its Linux container engine. Stop both services with
`docker compose down`.

## API contract

- `GET /health`: model, index, device, catalog, and label-use status.
- `GET /api/v1/guided-samples`: curated scenario metadata without labels.
- `POST /api/v1/match`: optional multipart image, optional title, and display `top_k`; at least one
  modality is required. Guided requests may include `query_posting_id` for self-exclusion.
- `POST /api/v1/match/batch`: up to eight independent image/title requests.
- `GET /api/v1/catalog/{posting_id}/image`: image evidence for a known catalog candidate.

The batch endpoint batches the user contract, not model execution; the GPU lock intentionally
serializes requests for deterministic, memory-bounded showcase behavior.

## Limitations

- The demo searches a fixed validation catalog; it is not a catalog-ingestion service.
- A newly uploaded listing can join an existing entity, but Phase 11 does not persist mutations.
- Approximate retrieval can omit a candidate, though Phase 7 measured a negligible validation
  recall difference at the selected HNSW setting.
- Calibration error and remaining false merges/splits mean review flags are operationally
  important.
- Authentication, rate limiting, monitoring, artifact distribution, and marketplace policy are
  outside this local portfolio demo.
