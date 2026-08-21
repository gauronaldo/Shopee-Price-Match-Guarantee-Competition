# Pretrained Representation Benchmark

Phase 9 status: **phase9_complete_validation_only**. This benchmark uses TorchVision EfficientNet-B1
`IMAGENET1K_V2` as a frozen image encoder. It performs no local training or fine-tuning and does
not access test.

## Protocol

- Split: validation, group-disjoint manifest inherited from Phases 1-8
- Retrieval: deterministic exact cosine over the full validation corpus
- Candidate budget: Top-50, identical to Phase 7
- Feature: 1,280-dimensional normalized penultimate EfficientNet-B1 representation
- Official preprocessing: resize 255, center crop 240, ImageNet mean/std normalization
- Weight SHA-256: `c27df63ce6eb17ef8bcea58922fd3a254cba910c720f41ee89d64d99fb7a4ddf`

## Quality comparison

| Validation system | Modalities | Pretraining | mAP@20 | Recall@20 | mAP@50 | Recall@50 |
|---|---|---|---:|---:|---:|---:|
| Scratch residual CNN | image | none | 0.53907 | 0.64667 | n/a | n/a |
| EfficientNet-B1 V2 | image | ImageNet-1K | 0.73753 | 0.82481 | 0.73934 | 0.88767 |
| Scratch multimodal joint | image + title | none | 0.87023 | 0.93780 | 0.87279 | 0.97438 |

## Efficiency

| Image representation | Feature params | Weight/checkpoint bytes | Embedding dim / bytes | Extraction listings/s | Exact p50 / p95 ms | Local training wall time |
|---|---:|---:|---:|---:|---:|---:|
| Scratch residual CNN | 3,060,000 | 36,876,216 | 256 / 3,512,320 | 270.50 | 0.442 / 0.656 | 13210.36 s |
| EfficientNet-B1 V2 | 6,513,184 | 31,600,329 | 1280 / 17,561,600 | 126.58 | 0.430 / 0.560 | 0 s locally* |

This is the modality-matched efficiency comparison. Both rows use the same validation listings and
exact-cosine protocol. The EfficientNet weight file is not a training checkpoint from this project.
Its local training cost is zero, but the external ImageNet pretraining cost is unknown and must not
be interpreted as free compute. The scratch multimodal system is omitted from this table because its
Phase 7 extraction benchmark used cached encoder outputs rather than end-to-end image/title
decoding.

## Recall by true group size

| Group size | Queries | Recall@50 |
|---|---:|---:|
| 2 | 1390 | 0.89712 |
| 3_to_5 | 1112 | 0.89951 |
| 6_to_9 | 457 | 0.89089 |
| 10_plus | 471 | 0.82875 |

## Failure evidence

- Top-1 points to a different label for `719` of
  `3430` queries (`20.96%`).
- No true group member appears in Top-50 for `190` queries
  (`5.54%`).
- Recall@50 falls to `0.82875`
  for groups of at least 10 listings, versus roughly `0.90` for smaller groups.

The bounded review sample includes same-category variant confusion and unrelated products with
similar composition. Those cases are consistent with a generic visual encoder learning category
and layout cues rather than exact identity-critical text, quantity, and model-number evidence.

## Interpretation

This comparison isolates the value of generic supervised image pretraining. It is directly fair
against the scratch CNN on data split, image modality, exact cosine ranking, and mAP@20/Recall@20.
The scratch multimodal row is a system-level ceiling rather than a modality-matched comparison
because it also uses title information.

ImageNet features can recognize shapes and semantic categories, but exact-product matching often
depends on packaging text, quantity, color, or model number. A pretrained image model is therefore
not expected to replace the multimodal pipeline automatically. Weak results are retained as a
measured domain-gap finding rather than tuned on test.

Model and transform contract: [official TorchVision EfficientNet-B1 documentation](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.efficientnet_b1.html).
Bounded top-1 errors and zero-recall examples remain in the ignored review artifact.

## Reproduction

```powershell
.venv\Scripts\python -m pip install -e ".[dev,retrieval,pretrained]"
.venv\Scripts\shopee-pretrained prepare-weights
.venv\Scripts\shopee-pretrained benchmark `
  --config configs\experiment\pretrained_image_benchmark.yaml
```
