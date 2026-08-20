# Scratch image encoder plan

## Phase 3 objective

Train and evaluate a compact PyTorch image encoder from random initialization for exact-product
retrieval. The experiment must use the frozen group-disjoint split, compare fairly with the Phase 2
image baselines, and explain failures involving visually similar variants and visually different
listings of the same product.

The first model is intentionally small: a residual CNN, global average pooling, a projection head,
and an L2-normalized listing embedding. No pretrained weights, pretrained image embeddings, or
multimodal evidence are allowed in this phase.

## Baselines and advancement target

- Supplied pHash validation mAP@20 / Recall@20: `0.2895 / 0.3174`.
- ORB validation mAP@20 / Recall@20: `0.6638 / 0.8284`.
- Primary advancement target: exceed ORB validation mAP@20 under the same retrieval protocol.
- Secondary targets: document Recall@1/5/10/20, embedding throughput, latency, memory, and model
  size.
- If the model does not beat ORB, stop and categorize the failure before increasing architecture
  size. A well-measured quality/latency trade-off may close the phase, but weak results must not be
  hidden.

## 1. Experiment contract and configuration

- [x] Add a model configuration for image size, residual stages, embedding dimension, and random
  initialization.
- [x] Add separate smoke, pilot, and full-training experiment configurations.
- [x] Record seed, split checksum, Git state, environment, hardware, hyperparameters, runtime, and
  checkpoint metadata for every run.
- [x] Reject pretrained checkpoints or unsupported model sources at configuration validation.
- [x] Define the primary validation metric and checkpoint-selection rule before training.
- [x] Keep test evaluation disabled until the architecture, checkpoint, and retrieval settings are
  frozen using train and validation only.

Suggested artifact names:

- `configs/model/scratch_residual_image_encoder.yaml`
- `configs/experiment/image_embedding_smoke.yaml`
- `configs/experiment/image_embedding_pilot.yaml`
- `configs/experiment/image_embedding_training.yaml`
- `reports/scratch_image_encoder_benchmark.md`

## 2. Image data pipeline

- [x] Implement a dataset that reads only manifest-selected listings and returns image, posting ID,
  and train-only label index.
- [x] Decode with OpenCV and convert BGR to RGB explicitly.
- [x] Implement deterministic validation/test preprocessing using aspect-preserving resize and pad.
- [x] Implement conservative train augmentation: resize/pad, mild crop, brightness/color changes,
  compression/noise, horizontal flip where appropriate, and small rotation.
- [x] Avoid augmentations that erase packaging text, change product color materially, or crop away
  quantity/model information.
- [x] Compute any normalization statistics from train only, or record a fixed non-pretrained
  normalization policy.
- [x] Add actionable handling for decode failures even though the current audit found none.
- [x] Verify deterministic sample order and preprocessing under the configured seed.

## 3. Product-aware sampling

- [x] Implement a deterministic `P x K` batch sampler using train `label_group` values only.
- [x] Support groups with only two listings and document whether sampling uses replacement.
- [x] Ensure every metric-learning batch contains positive pairs and multiple negative groups.
- [x] Test epoch length, batch composition, seed reproducibility, and no validation/test IDs.
- [x] Log the number of unique groups and listings observed per epoch.

## 4. Residual image encoder

- [x] Implement convolution, normalization, activation, residual block, downsampling, and global
  average pooling directly in the repository using PyTorch modules.
- [x] Add a configurable projection head and L2-normalized output embedding.
- [x] Use explicit random initialization and record the initialization policy.
- [x] Keep the forward pass free of file-system and logging side effects.
- [x] Report parameter count, tensor shapes by stage, embedding dimension, and serialized size.
- [x] Test output shape, finite values, unit embedding norm, gradient flow, and serialization parity.

## 5. Loss and training loop

- [x] Implement supervised contrastive loss as the first objective.
- [x] Test the loss against a hand-checkable toy example and confirm finite gradients.
- Optional follow-up: mixed precision can be benchmarked later if extraction or training cost
  becomes a bottleneck; it is not required for the Phase 3 quality claim.
- [x] Implement optimizer, scheduler, gradient clipping, checkpointing, early stopping, and resume
  behavior through configuration.
- [x] Save checkpoints atomically with config, seed, split checksum, epoch, metric, and model shape
  metadata.
- [x] Log train loss, validation retrieval metrics, learning rate, epoch time, and peak memory in a
  structured JSON/CSV run record.
- [x] Treat batch-hard triplet loss as an ablation only after supervised contrastive training is
  stable; do not begin offline hard-negative mining from Phase 6.

## 6. Mandatory smoke gates

- [x] Run all existing unit and integration tests.
- [x] Load a deterministic fixture batch and verify preprocessing numerically.
- [x] Run one forward and backward optimization step.
- [x] Confirm every intended parameter receives a finite gradient.
- [x] Overfit one tiny batch until retrieval becomes hand-checkably correct.
- [x] Overfit a small multi-batch synthetic subset through the integration pipeline.
- [x] Confirm checkpoint serialization and deterministic evaluation preprocessing parity.
- [x] Verify exact-search metrics on a tiny fixture with known neighbors.
- [x] Estimate full-run duration and checkpoint/storage requirements before pilot training.

No pilot or full training may start until every smoke gate passes.

## 7. Bounded pilot experiment

- [x] Train on a bounded, deterministic subset large enough to contain varied group sizes.
- [x] Inspect learning curves for divergence, collapse, overfitting, and ineffective augmentation.
- [x] Measure embedding-norm distribution and pairwise similarity distributions for positives and
  negatives.
- [x] Evaluate exact validation retrieval at K = 1, 5, 10, and 20.
- [x] Manually inspect nearest neighbors for at least the established error-analysis categories.
- [x] Change only one major factor at a time and record the result as an ablation.
- [x] Approve one frozen full-training configuration based only on pilot/train/validation evidence.

## 8. Full training and checkpoint selection

- [x] Run the frozen full-training configuration and preserve the selected artifacts.
- [x] Select the checkpoint using the declared validation metric only.
- [x] Record total training time, hardware, peak memory, convergence epoch, and checkpoint size.
- [x] Confirm the selected checkpoint loads and reproduces its saved validation metric.
- Additional seeds are deferred to the final repeated-seed evaluation if compute permits; the
  current result is reported as a single fixed-seed experiment.
- [x] Freeze the checkpoint and retrieval configuration before the single test evaluation.

## 9. Retrieval evaluation and efficiency

- [x] Extract one normalized embedding per validation and frozen test listing.
- [x] Use exact cosine search first; FAISS remains a later-phase optimization.
- [x] Report mAP@20, Recall@1/5/10/20, Precision@K, hit rate, and mean per-query F1 where compatible.
- [x] Compare against pHash and ORB in the same table and split protocol.
- [x] Measure embedding extraction throughput, p50/p95 query latency, peak CUDA memory, checkpoint size,
  and embedding storage size.
- [x] Report metrics by group size and exact-positive-pHash status.
- [x] Separate embedding-extraction cost from nearest-neighbor search cost.

## 10. Structured error analysis

- [x] Generate a bounded local-only nearest-neighbor review manifest with no images committed.
- [x] Tag failures using the repository taxonomy: crop/overlay, different-image same-product,
  same-brand variant, quantity/size/color/model conflict, poor image quality, packaging redesign,
  questionable label, and retrieval miss.
- [x] Quantify each observed category instead of presenting only selected examples.
- [x] Compare the scratch encoder with pHash and the candidate-assisted ORB pipeline, and document
  the remaining image-only failure categories.
- [x] Keep review evidence image-only without using title features to rescue failures.
- [x] Record the next experiment only when it targets a measured failure category.

## Phase 3 exit criteria

- [x] Architecture, tensor-shape, normalization, gradient, sampler, loss, and serialization tests
  pass.
- [x] Tiny-batch and tiny-subset overfit gates pass.
- [x] Training is stable under the frozen configuration and seed.
- [x] The selected scratch checkpoint is evaluated independently as an image-only retriever.
- [x] Comparison with pHash and ORB uses the frozen split and records the ORB candidate-protocol
  caveat.
- [x] The model beats ORB validation mAP@20, or a clearly measured quality/latency trade-off and
  categorized failure analysis explains why it does not.
- [x] Nearest-neighbor failures are manually reviewed and recorded.
- [x] Final comparison, frozen-test evaluation, and categorized failure reports are recorded.
- [x] No raw images, local galleries, checkpoints, caches, or oversized outputs are tracked.
- [x] Phase 3 results and limitations are recorded before Phase 4 begins.

Phase 3 is closed. The next modeling phase is the independent scratch text encoder; multimodal
fusion remains gated until that text model is evaluated on its own.
