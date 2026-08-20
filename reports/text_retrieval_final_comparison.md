# Text retrieval final comparison

## Decision

Phase 4 is complete. The scratch character TextCNN is a valid, independently trained text
encoder, but character TF-IDF remains the stronger standalone title retriever under the same
group-disjoint split and exact Top-20 protocol. The neural encoder is therefore retained as a
learned modality for multimodal fusion, not presented as a replacement for the classical text
baseline.

## Frozen evaluation

The TextCNN checkpoint was selected by validation mAP@20. Its checkpoint, training configuration,
training metrics, validation pair threshold, and retrieval protocol were then locked before one
held-out test evaluation.

- Checkpoint SHA-256: `cbb77e4d76c1909c24b0e30654eacde2a0f752bb5d2ba795d45b01ec1189f7c1`
- Training config SHA-256: `dbd9ac71a42f0dec92412c8aabbc3e987e71356ad6d8d0a01230eebcc93caf77`
- Training metrics SHA-256: `e92bd3704d09e6e3ae36e89c2d219c69e7b3d7daa6ad70e856ae0e0e73095030`
- Selected epoch: `26` of `30`
- Frozen validation pair threshold: `0.654422`

## Retrieval comparison

Both methods use title text only. TF-IDF fits character n-gram vocabulary and IDF statistics on
train; TextCNN fits a 40-character vocabulary on train and learns a 256-dimensional normalized
embedding from random initialization. All values below are macro-averaged per query.

| Method | Validation mAP@20 | Validation Recall@20 | Test mAP@20 | Test Recall@20 |
|---|---:|---:|---:|---:|
| Character TF-IDF | **0.8635** | **0.9385** | **0.8564** | **0.9291** |
| Scratch character TextCNN | 0.7570 | 0.8741 | 0.7484 | 0.8698 |
| TextCNN minus TF-IDF | -0.1065 | -0.0644 | -0.1080 | -0.0593 |

TextCNN generalizes consistently: validation-to-test mAP@20 changes by only `-0.0086`, and
Recall@20 changes by `-0.0044`. Its lower absolute quality is therefore not explained by a bad
test split or severe validation overfitting. The result instead shows that exact character n-gram
overlap is exceptionally strong for this dataset, while the compact global TextCNN embedding
loses some sparse model-number, quantity, spelling, and long-tail lexical evidence.

## Pair decision comparison

| Method | Validation pair F1 | Test pair F1 |
|---|---:|---:|
| Character TF-IDF | not retained in the aggregate report | **0.7048** |
| Scratch character TextCNN | **0.6030** | 0.5619 |

The TextCNN test pair F1 is `0.1429` below TF-IDF. Its precision remains stable from validation
to test (`0.6540` to `0.6510`), but recall falls from `0.5593` to `0.4942` at the frozen validation
threshold. This makes the learned text score useful as complementary evidence, but insufficient
as the sole final match decision.

## Failure-analysis interpretation

Manual validation review separates two major error families. Among 40 high-confidence Top-1
false matches, 42.5% are same-category or same-brand listings with a different model, variant, or
quantity. Among 40 complete Top-20 retrieval misses, questionable labels, broad groups, or label
fragmentation account for 30.0%; lexical/paraphrase/multilingual gaps account for 25.0%; and
model, variant, quantity, or compatibility distinctions account for 22.5%.

These findings motivate Phase 5 directly: preserve TF-IDF as a strong lexical channel, combine
the scratch TextCNN with the frozen scratch image encoder, and measure whether multimodal evidence
recovers lexical misses without increasing variant-confusion errors. Detailed validation-only
examples and taxonomy are recorded in
[`scratch_text_encoder_failure_analysis.md`](scratch_text_encoder_failure_analysis.md).

## Phase 4 closure

The Phase 4 exit criteria are satisfied:

- training is stable and reproducible across all 30 configured epochs;
- vocabulary fitting is train-only and preserves digits and unit-bearing text;
- validation and one frozen held-out test evaluation are recorded independently;
- the model is compared fairly with character TF-IDF;
- categorized validation failures are documented;
- the weak result relative to TF-IDF is retained and explained rather than hidden.

