# Custom text encoder

## Outcome

Phase 4 implements a train-only character TextCNN initialized randomly. It reaches validation/test
mAP@20 of `0.75698 / 0.74841` and Recall@20 of `0.87414 / 0.86978`. The learned encoder is stable
and useful for multimodal fusion, but character TF-IDF remains the stronger standalone title
retriever.

## System design

- NFKC/case-fold normalization that preserves digits, units, punctuation-bearing model tokens,
  and multilingual characters.
- Deterministic train-only vocabulary with explicit PAD/UNK symbols.
- Random character embeddings, convolution kernels 3/5/7, global max pooling, projection head,
  and normalized 256-dimensional title embeddings.
- Product-aware `P x K` batches, supervised contrastive loss, validation-only checkpoint and
  threshold selection, exact cosine retrieval, and length/OOV diagnostics.

The model has `455,040` parameters, vocabulary size `40`, validation unknown-character rate `0`,
and only about `0.50%` of validation titles exceed the 128-character limit.

## Experiment progression

| Stage | Validation mAP@20 | Validation Recall@20 | Pair F1 | Decision |
|---|---:|---:|---:|---|
| Smoke, four updates | 0.64049 | 0.73099 | 0.55890 | Engineering gate only |
| 12-epoch pilot | 0.71376 | 0.82911 | 0.56102 | Architecture promoted unchanged |
| Full validation | **0.75698** | **0.87414** | **0.60295** | Epoch 26 selected |
| Frozen test | **0.74841** | **0.86978** | **0.56187** | One-time evaluation |

The pilot improved consistently and showed clear positive/negative cosine separation
(`0.63948 / 0.00526`). Long normalized titles remained the weakest length band, motivating error
analysis rather than immediate model expansion.

## Full-training trend

| Epoch | Train loss | Validation mAP@20 |
|---:|---:|---:|
| 0 | 0.58699 | 0.65048 |
| 3 | 0.21229 | 0.70350 |
| 6 | 0.09898 | 0.71375 |
| 9 | 0.07395 | 0.72992 |
| 12 | 0.04760 | 0.73445 |
| 15 | 0.04017 | 0.73986 |
| 18 | 0.03150 | 0.74458 |
| 21 | 0.02562 | 0.75032 |
| 25 | 0.02241 | **0.75698** |
| 29 | 0.02003 | 0.75679 |

The saved checkpoint is reported as epoch 26 in the run metadata while the zero-based history row
with the best `0.75698` value is index 25. This is a reporting-index convention, not a metric
disagreement.

## TF-IDF comparison

| Method | Validation mAP@20 | Validation Recall@20 | Test mAP@20 | Test Recall@20 |
|---|---:|---:|---:|---:|
| Character TF-IDF | **0.8635** | **0.9385** | **0.8564** | **0.9291** |
| Custom character TextCNN | 0.7570 | 0.8741 | 0.7484 | 0.8698 |
| TextCNN minus TF-IDF | -0.1065 | -0.0644 | -0.1080 | -0.0593 |

TF-IDF retains sparse model-number, quantity, spelling, and long-tail n-gram evidence better than
the compact global TextCNN embedding. TextCNN validation-to-test mAP changes by only `-0.0086`, so
the gap is not explained by severe overfitting.

At frozen threshold `0.654422`, test pair precision/recall/F1 are
`0.65102 / 0.49420 / 0.56187`; TF-IDF test pair F1 is `0.7048`.

## Failure analysis

The bounded validation review contains 40 Top-1 false matches and 40 complete Top-20 misses.

| Top-1 false-match category | Count | Share |
|---|---:|---:|
| Same category/brand, different model, variant, or quantity | 17 | 42.5% |
| Paraphrase, abbreviation, lexical mismatch, or weak positive title | 11 | 27.5% |
| Probable cross-label duplicate/fragmentation | 7 | 17.5% |
| Questionable or overly broad ground-truth group | 4 | 10.0% |
| Long seller/shipping noise | 1 | 2.5% |

| Top-20 miss category | Count | Share |
|---|---:|---:|
| Questionable, broad, or commercially inconsistent label | 12 | 30.0% |
| Lexical, paraphrase, or multilingual gap | 10 | 25.0% |
| Model, variant, quantity, or compatibility confusion | 9 | 22.5% |
| Short or underspecified title | 8 | 20.0% |
| Long seller/shipping noise | 1 | 2.5% |

These results justify keeping digits, units, quantities, and model conflicts visible to later pair
scoring and using image evidence when titles are short or semantically inconsistent.

## Efficiency and frozen evidence

- Validation/test embedding throughput: `16,609.75 / 10,741.39` listings/s.
- Test exact-ranking p50/p95: `0.343 / 0.425 ms/query`.
- Checkpoint SHA-256: `cbb77e4d76c1909c24b0e30654eacde2a0f752bb5d2ba795d45b01ec1189f7c1`
- Canonical training-config SHA-256: `7fac21a3c2a45fdfa358da3353ec053aa96edea1554eea3da80abacba16de844`
- Training-metrics SHA-256: `e92bd3704d09e6e3ae36e89c2d219c69e7b3d7daa6ad70e856ae0e0e73095030`

The checkpoint, threshold, vocabulary, and Top-20 protocol were frozen before the single test
evaluation. Phase 4 is closed.
