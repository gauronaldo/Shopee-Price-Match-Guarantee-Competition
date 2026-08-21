# Scratch text encoder plan

## Phase 4 objective

Train and evaluate an independent title encoder from random initialization. The model uses only
titles and training-split labels, fits its character vocabulary on train only, and reserves the
test split until architecture, checkpoint, and retrieval protocol are frozen.

## Initial method

- NFKC and case-fold title normalization shared with the classical baseline.
- Identity-critical digits and unit characters are preserved.
- Train-only character vocabulary with explicit PAD and UNK tokens.
- Fixed-length padded sequences with deterministic truncation diagnostics.
- Random character embeddings and a multi-kernel TextCNN.
- Global max pooling, projection head, and L2-normalized 256-dimensional embedding.
- Product-aware `P × K` batches and supervised contrastive loss.
- Exact full-split cosine retrieval selected by validation mAP@20.

Character modeling is selected over a word vocabulary because marketplace titles contain noisy
spelling, multiple languages, abbreviations, concatenated model codes, quantities, and units. The
first experiment intentionally avoids pretrained tokenizers and word embeddings.

## Leakage and evaluation contract

- Vocabulary frequency and token IDs are fitted from training titles only.
- Validation uses the frozen group-disjoint split for checkpoint and threshold selection.
- The training command cannot evaluate test.
- The Phase 2 character TF-IDF validation mAP@20 `0.8635` is the primary text baseline.
- Results are stratified by group size and normalized title length.
- Unknown-character and truncation rates are recorded explicitly.

## Gates

- [x] Configuration rejects pretrained checkpoints and test evaluation during training.
- [x] Normalization, vocabulary, padding, unknown-token, model-shape, norm, and gradient tests.
- [x] Deterministic product-aware batches and supervised contrastive loss are reused.
- [x] Configuration-driven checkpointing, progress logging, exact retrieval, and report output.
- [x] Run real-data smoke training and inspect vocabulary/OOV/truncation diagnostics.
- [x] Run a bounded pilot and verify stable learning before defining a full configuration.
- [x] Freeze the selected checkpoint and compare it independently with character TF-IDF.
- [x] Categorize failures involving brand, quantity/unit, model number, typo, and multilingual text.

## Closure

Phase 4 is complete. The frozen scratch TextCNN reached validation/test mAP@20 of
`0.75698 / 0.74841` and Recall@20 of `0.87414 / 0.86978`. Character TF-IDF remains stronger at
test mAP@20 `0.8564`, so Phase 5 should treat the learned embedding as complementary multimodal
evidence and retain TF-IDF as a lexical reference channel. See
[`../reports/text_encoder.md`](../reports/text_encoder.md).
