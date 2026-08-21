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
    G --> J[Exact retrieval reference]
    J --> K[FAISS HNSW comparison]
    H --> L[Validation-frozen graph policy]
    J --> L
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
    D --> E[Exact or FAISS Top-K retrieval]
    E --> F[Deduplicate candidate pairs]
    F --> G[Symmetric pair probabilities]
    G --> H{Probability and reciprocal-rank gates}
    H -->|reject| I[No graph edge]
    H -->|accept| J{Variant and component consistency}
    J -->|reject| K[Blocked merge or manual review]
    J -->|accept| L[Union-find component merge]
    L --> M[Entity ID, confidence, review flag]
    I --> M
    K --> M
```

Candidate retrieval controls recall: missing candidates cannot be recovered downstream. Pair
scoring controls local match evidence. Graph consistency controls entity-level risk because one
false-positive edge can transitively merge otherwise correct groups.

## Online evidence contract

For one query listing, the same encoders produce a joint embedding, retrieve Top-K catalog
candidates, and score candidate pairs. The response should expose candidate IDs, total confidence,
image/title similarity evidence, predicted entity or no-confident-match state, and a manual-review
flag. No deployment claim is made until the final demo is independently tested.
