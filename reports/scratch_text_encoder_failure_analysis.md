# Scratch text encoder categorized failure analysis

## Scope

This analysis uses the deterministic validation-only review manifest generated from the selected
full-training checkpoint. The local manifest contains 40 top-1 false matches, 40 Top-20 retrieval
misses, and 40 top-1 successes. Titles, posting IDs, and manual-review records remain under ignored
`artifacts/`; only aggregate categories and sanitized examples are committed.

The review is bounded and deterministic rather than statistically representative. Percentages
below describe the inspected samples only and must not be extrapolated to the entire validation
split.

## Top-1 false-match taxonomy

| Primary category | Count | Share | Interpretation |
|---|---:|---:|---|
| Same category or brand, different model/variant/quantity | 17 | 42.5% | Shared product words dominate identity-critical differences. |
| Lexical mismatch, paraphrase, abbreviation, or weak positive title | 11 | 27.5% | The true pair uses different wording or provides too little lexical evidence. |
| Probable cross-label duplicate or label fragmentation | 7 | 17.5% | The predicted pair appears to describe the same product despite different labels. |
| Questionable or overly broad ground-truth group | 4 | 10.0% | The labeled positives themselves appear commercially inconsistent or overly generic. |
| Long seller/shipping noise | 1 | 2.5% | Non-product terms become stronger than the underlying identity evidence. |

Frequent model/variant confusions include two football-shoe models from the same brand, different
smartwatch models, phone cases with overlapping compatibility lists, different soft-lens
solutions, and generic products with a changed volume or pack size. These errors show why digits,
units, and model codes must remain explicit inputs to the later pair head instead of relying only
on a pooled title embedding.

Several false matches are likely label fragmentation rather than clear model errors. Examples
include almost identical rose-water titles, small-tip cotton swabs with the same 100-piece count,
the same skincare starter pack, and nearly identical branded children's lap desks. They remain
counted as errors because evaluation labels are not rewritten.

## Top-20 retrieval-miss taxonomy

| Primary category | Count | Share | Interpretation |
|---|---:|---:|---|
| Questionable, broad, or commercially inconsistent label | 12 | 30.0% | Positive titles provide conflicting product identity evidence. |
| Lexical/paraphrase/multilingual gap | 10 | 25.0% | Same-product titles share meaning but few character sequences. |
| Model, variant, quantity, or compatibility confusion | 9 | 22.5% | Similar alternatives occupy Top-20 ahead of the exact product. |
| Short or underspecified query title | 8 | 20.0% | A brand/model fragment is insufficient to retrieve its detailed counterpart. |
| Long seller/shipping noise | 1 | 2.5% | Marketplace boilerplate overwhelms the product terms. |

Short-title misses include bare model or collection names whose positive listing adds RAM/storage,
product type, or brand context. Paraphrase misses include book-safe versus dictionary-safe wording,
different Indonesian/English names for the same kitchen tool, and abbreviated fashion/product
terms. These are natural limitations of a local character CNN without pretrained semantic
knowledge.

## Quantitative context

- Validation mAP@20: `0.75698`.
- Validation Recall@20: `0.87414`.
- Validation hit rate@20: `0.94927`.
- Validation pair F1 at the selected threshold: `0.60295`.
- Title length 31–60 mAP@20: `0.78495`.
- Title length 101+ mAP@20: `0.64954`.
- Group size 6–9 mAP@20: `0.71867`, the weakest group-size band.

Only about `0.50%` of validation titles exceed the 128-character limit, so truncation alone cannot
explain the full long-title gap. Long titles also contain compatibility lists, promotional text,
shipping terms, and repeated category keywords that create hard negatives.

## Decision

The measured failures do not justify enlarging the text encoder before multimodal fusion. The
highest-value next mechanisms are already planned:

1. preserve the frozen TextCNN as an independently measured learned representation;
2. retain TF-IDF as a strong lexical candidate/evidence channel;
3. expose digits, units, quantities, and model-token conflicts to the Phase 5 pair/fusion head;
4. use image evidence when titles are short, paraphrased, or commercially ambiguous;
5. revisit hard negatives and label ambiguity in Phase 6 rather than tuning on held-out test.

The local review manifest is validation-only. No test example was inspected or categorized during
this analysis.

