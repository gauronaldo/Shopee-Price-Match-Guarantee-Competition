# Problem definition and identity policy

## Objective

Resolve Shopee listings into the same underlying purchasable product using image and title
evidence. The system performs candidate retrieval, pair matching, and conservative clustering;
these are separate stages with separate failure modes and metrics.

The Kaggle `label_group` is the operational ground truth for experiments. It may contain noise or
ambiguity, so suspicious examples must be documented rather than silently relabeled.

## Definition of “same product”

Two listings are the same product when a buyer intending to purchase one exact SKU could receive
the other without an identity-relevant substitution. Evidence should agree on brand/product
family and all available variant-defining attributes.

Identity-preserving differences can include:

- seller wording, spelling, language, punctuation, or title order;
- crop, background, lighting, overlay, compression, or viewpoint;
- seller identity or listing price;
- packaging redesign when model, variant, quantity, and included contents remain equivalent.

Identity-changing differences include:

- model or part number;
- size, dimensions, fit, color, flavor, scent, or formulation;
- capacity, weight, volume, count, pack size, or bundle composition;
- included accessories, edition, region/compatibility, or condition when those change the SKU.

When these attributes are missing, contradictory, or unreadable, the system should lower
confidence or request manual review. Visual resemblance alone is insufficient.

## Dataset assumptions

- `posting_id` uniquely identifies a listing.
- `label_group` represents one product identity and is never split across train/validation/test.
- `image_phash` is useful evidence, not a ground-truth match rule.
- Titles are noisy and multilingual; digits and units are identity-critical.
- A listing has one referenced image in the competition data.
- Data audit may invalidate any assumption above; Phase 1 records measured facts separately.

## Online contract

Input: one decodable listing image and one non-empty title.

Output:

- ordered Top-K candidate posting IDs;
- per-candidate match confidence in `[0, 1]`;
- image and title similarity in `[0, 1]` as distinct evidence;
- `match`, `no_match`, or `needs_review` decision per candidate;
- predicted duplicate group or `null` when no confident group exists;
- result-level manual-review flag and version metadata.

The system must support “no confident match”; Top-K retrieval does not imply a positive match.

## Batch contract

Input: a catalog of unique posting IDs with decodable images and non-empty titles.

Output:

- versioned candidate pairs and pair probabilities;
- one final entity/cluster ID per listing;
- cluster confidence and member IDs;
- review reasons for ambiguous or potentially over-merged clusters.

Pair probability and cluster confidence are not interchangeable. Thresholds are selected on
validation data only and frozen before final test evaluation.

## Explicit non-goals

- Generic visual similarity or category classification.
- Price, seller, counterfeit, policy, or fraud decisions.
- Manual correction of competition labels without a versioned adjudication process.
- Leaderboard optimization that compromises group-disjoint evaluation.
- UI, API, deployment, pretrained comparison, or large training during early phases.

## Success criteria

The final project reports retrieval, pair-classification, clustering, efficiency, and calibration
metrics. Results are stratified by group size and relevant difficulty buckets. Every claim names
the split manifest, configuration, seed, environment, and saved run. False merges receive special
attention because one bad graph edge can combine large unrelated groups.

