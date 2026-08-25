# System Architecture

## Training and evidence flow

```mermaid
flowchart LR
    A[Group-disjoint catalog split] --> B[Custom residual image encoder]
    A --> C[Custom character TextCNN]
    B --> D[Normalized image embedding]
    C --> E[Normalized title embedding]
    D --> F[Residual multimodal fusion]
    E --> F
    F --> G[Normalized joint embedding]
    G --> H[Symmetric pair head]
    A --> I[Train-only hard-negative mining]
    G --> I
    I --> H
    G --> J[Dense cosine retrieval]
    A --> K[Train-fitted character TF-IDF and pHash]
    J --> N[Weighted reciprocal-rank fusion]
    K --> N
    J --> O[FAISS HNSW comparison]
    H --> L[Validation-frozen graph policy]
    N --> L
```

The image and text encoders are trained from random initialization. Fusion starts only after both
unimodal checkpoints are frozen. Hard-negative mining changes the pair head without changing the
joint retrieval embedding. Validation selects checkpoints and operating rules; test is descriptive
only.

## Batch entity-resolution flow

```mermaid
flowchart TD
    A[Catalog images and titles] --> B[Validate and preprocess]
    B --> C[Image and title embeddings]
    C --> D[512-dimensional joint embedding]
    B --> E[Character TF-IDF and pHash evidence]
    D --> F[Dense Top-50 retrieval]
    E --> G[Sparse and hash candidates]
    F --> H[Weighted RRF Top-75]
    G --> H
    H --> I[Deduplicate candidate pairs]
    I --> J[Symmetric pair probabilities]
    J --> K{Probability and reciprocal-rank gates}
    K -->|reject| L[No graph edge]
    K -->|accept| M{Variant and component consistency}
    M -->|reject| N[Blocked merge or manual review]
    M -->|accept| O[Union-find component merge]
    O --> P[Entity ID, confidence, review flag]
    L --> P
    N --> P
```

Candidate retrieval controls recall: missing candidates cannot be recovered downstream. Pair
scoring controls local match evidence. Graph consistency controls entity-level risk because one
false-positive edge can transitively merge otherwise correct groups.

## Online evidence contract

For one query listing, the same encoders produce a joint embedding, retrieve Top-K catalog
candidates, and score candidate pairs. The response should expose candidate IDs, total confidence,
image/title similarity evidence, predicted entity or no-confident-match state, and a manual-review
flag.

```mermaid
flowchart LR
    A[Uploaded image and title] --> B[FastAPI validation]
    B --> C[Custom image and text encoders]
    C --> D[Residual multimodal fusion]
    D --> E[FAISS HNSW Top-50 retrieval]
    E --> F[Symmetric pair head]
    F --> G[Reciprocal and variant gates]
    G --> H[Entity, evidence, and review flag]
    H --> I[Streamlit presentation]
```

The service loads all models and the index once at startup. Its showcase catalog is the validation
split, not the held-out test split. Ground-truth labels remain outside the request path. The API and
UI are portfolio demonstration components, not a claim of production readiness.
