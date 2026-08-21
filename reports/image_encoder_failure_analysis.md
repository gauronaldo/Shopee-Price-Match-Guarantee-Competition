# Scratch image encoder categorized failure analysis

## Scope and sampling

This analysis covers the deterministic review manifest emitted by the final validation run. The
manifest contains 20 top-1 false matches, 20 Top-20 retrieval misses, and 20 top-1 successes. Raw
listing IDs, image paths, and local contact sheets remain under ignored `artifacts/` and
`.scratch/`; no competition image is committed.

The samples are the first bounded examples selected by the deterministic review generator, not a
random or statistically representative sample. Percentages below describe this review set only
and must not be extrapolated to the complete validation split.

## Top-1 false-match taxonomy

| Primary category | Count | Share | Interpretation |
|---|---:|---:|---|
| Shared layout, background, color block, or coarse silhouette | 14 | 70% | The encoder overweights merchandising composition and global shape. |
| Coarse-category hard negative | 4 | 20% | The broad category is plausible, but the exact purchasable product differs. |
| Probable label fragmentation | 2 | 10% | Image and title evidence strongly suggest the cross-label pair may be the same product. |

Observed layout shortcuts include rectangular packages, centered dark objects on white, dense
catalog collages, apparel model poses, and repeated colorful textures. Examples include a vitamin
box matched to soap packaging, a liquid-soap advertisement matched to a black bike computer, and
plastic flowers matched to a plated food image through clustered color and texture.

The coarse-category cases were different noodle products, football shoes versus wedge sandals,
a dress versus a T-shirt, and two different shirt products. These errors show category recognition
without exact-product resolution.

Two pairs are probable competition-label fragmentation rather than clear commercial errors: two
Napolly character-table listings share nearly identical imagery and cosine `0.995`, while two
Pigeon small-tip cotton-swab listings have near-identical titles/images and cosine `0.968`. They
remain counted as errors because labels are never rewritten during evaluation.

## Retrieval-miss taxonomy

| Primary category | Count | Share | Interpretation |
|---|---:|---:|---|
| Shared layout, background, color block, or coarse silhouette | 15 | 75% | Irrelevant visual neighbors occupy Top-20 while the true product is not retrieved. |
| Category/brand-family hard negative | 5 | 25% | Similar apparel or same-brand accessories displace the exact product. |

The category/brand-family misses include different dresses and shirts, plus a Baseus car charger
whose nearest result is a different Baseus charging cable. The latter is especially relevant to
entity resolution: brand evidence is correct, but product function and model identity differ.

## Quantitative context

- Validation top-1 hit rate: `0.60904`.
- Validation Top-20 hit rate: `0.77318`.
- Validation mAP@20 with an exact-pHash positive: `0.83327`.
- Validation mAP@20 without an exact-pHash positive: `0.43893`.
- Test mAP@20 without an exact-pHash positive: `0.46808`.

The scratch model learns substantially beyond exact duplicates, but the approximately 0.39
validation mAP gap between the exact-positive and no-exact-positive strata confirms that visually
different same-product listings remain the central image-only weakness.

## Decision

Increasing epochs is not the next evidence-based response: validation mAP improved only `0.00314`
over the final five epochs while the learning rate reached `1e-6`. The measured failures instead
motivate the independently planned scratch text encoder and later multimodal fusion, where digits,
brands, quantities, units, and model tokens can reject visually plausible but identity-incorrect
neighbors. Architectural expansion of the image encoder is deferred until multimodal ablation can
show whether these image-only errors remain the system bottleneck.

## Limitations

- Categories are subjective manual labels on a bounded sample.
- A single primary category is assigned even when several mechanisms may apply.
- The validation manifest's `manual_category` fields remain local; this report contains only
  aggregate counts and sanitized descriptions.
