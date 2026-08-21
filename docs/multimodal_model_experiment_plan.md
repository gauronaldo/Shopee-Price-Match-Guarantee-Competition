# Custom multimodal model experiment plan

## Phase 5 objective

Learn a joint listing representation from the completed scratch image and text systems without
introducing external pretrained weights. The first experiment freezes both source encoders,
caches their deterministic train/validation embeddings, and trains only a repository-owned fusion
projection and symmetric pair head.

## Initial method

- Frozen 256-dimensional scratch image embedding from Phase 3.
- Frozen 256-dimensional scratch text embedding from Phase 4.
- Simple validation-tuned score fusion as a non-learned multimodal reference.
- Learned features `[v, t, v*t, |v-t|]` projected to a normalized joint embedding.
- Score-preserving residual variant initialized to the selected simple-fusion geometry.
- Symmetric pair features `[z1*z2, |z1-z2|]` and a binary pair head.
- Product-aware batches, supervised contrastive loss, and balanced pair BCE.
- Exact full-validation retrieval; the held-out test split remains disabled.

Caching is deliberate: frozen CNN/TextCNN inference is performed once per listing, while the
lightweight fusion module can be trained repeatedly without re-decoding images or recomputing base
embeddings. Every cache records source checkpoint, data, split, preprocessing, and tokenization
fingerprints.

## Required comparisons

- image only;
- text only;
- simple score fusion;
- learned multimodal fusion;
- pair head off/on;
- contrastive and pair-loss components, after the initial smoke gate.

## Gates

- [x] Frozen source configs and checkpoints are SHA-256 verified.
- [x] Configuration rejects test evaluation and encoder unfreezing in the initial experiment.
- [x] Model shape, normalization, symmetry, gradient, and deterministic pair-sampling tests pass.
- [x] Frozen embedding cache reproduces Phase 3 and Phase 4 validation metrics within `1e-4`.
- [x] A real-data smoke run completes with all five validation comparisons.
- [x] A bounded pilot and pair-loss ablation justify the full Phase 5 configuration.
- [x] Full validation-only training runs with early stopping and test disabled.
- [x] Categorize modality-disagreement and pair-reranking failures.
- [x] Run three deterministic seeds and record mean/standard deviation.
- [x] Run contrastive-only and pair-BCE-only validation ablations.
- [x] Freeze the selected artifacts and protocol before one-time test evaluation.
- [x] Evaluate the held-out test exactly once and refuse output overwrite.

## Pilot decision

The projected pilot reached mAP@20 `0.81244`, improving over frozen text but remaining below simple
fusion `0.87358`. Pair-BCE weights `0.50`, `0.25`, and `0.10` were then compared with all other
inputs fixed. Weight `0.10` led validation mAP@20 and pair F1, so it was selected for full training.
The full run selected epoch 1 at mAP@20 `0.87903`, Recall@20 `0.93780`, and pair F1 `0.71285`, then
stopped after 7 of 30 epochs. See
[`../reports/multimodal_fusion_training_summary.md`](../reports/multimodal_fusion_training_summary.md).

Phase 5 is closed. The canonical held-out result is mAP@20 `0.86848`, Recall@20 `0.93235`, and
pair F1 `0.68429`. See
[`../reports/multimodal_model_final_comparison.md`](../reports/multimodal_model_final_comparison.md).
