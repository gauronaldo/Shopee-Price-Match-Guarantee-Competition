# Pretrained Representation Benchmark Plan

## Objective

Measure the value and limitations of generic supervised image pretraining after the scratch system
and entity-resolution pipeline are complete. The first Phase 9 benchmark uses the official
TorchVision EfficientNet-B1 `IMAGENET1K_V2` representation without local training or fine-tuning.

## Fair-comparison contract

- use the same group-disjoint validation split as Phases 3-8;
- use the full validation corpus and deterministic exact cosine search;
- report mAP@20 and Recall@20 against the scratch image encoder;
- also report Top-50 metrics under the frozen Phase 7 candidate budget;
- keep the scratch multimodal result visible but label it as a system-level, not modality-matched,
  comparison;
- do not select weights, architecture, preprocessing, K, or thresholds on test;
- do not access test in Phase 9 model selection.

## Frozen representation

The classifier is removed after loading the SHA-256-verified official ImageNet-1K V2 weight. The
1,280-dimensional penultimate features are L2-normalized. Preprocessing uses the transform bundled
with the weight enum: resize to 255, center crop to 240, rescale to `[0, 1]`, and apply ImageNet
mean/std normalization.

The benchmark records feature parameters, weight size, embedding storage, extraction throughput,
exact-search latency, retrieval metrics, group-size recall, and bounded failure examples. Local
training cost is zero; the unknown external ImageNet pretraining cost must not be reported as zero.

## Exit criteria

- weight origin and SHA-256 are recorded;
- preprocessing and feature dimension match the official weight contract;
- exact retrieval uses the frozen Phase 7 K and validation split;
- quality and efficiency are compared with scratch evidence in one report;
- domain-gap failures are documented honestly;
- tests and static checks pass;
- test remains untouched.

## Manual command

```powershell
.venv\Scripts\python -m pip install -e ".[dev,retrieval,pretrained]"
.venv\Scripts\shopee-pretrained prepare-weights
.venv\Scripts\shopee-pretrained benchmark `
  --config configs\experiment\pretrained_image_benchmark.yaml
```
