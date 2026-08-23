# Data Quality and Leakage-Safe Split

This report is generated from aggregate statistics only. Raw Kaggle CSVs and images remain local.

## Audit protocol

- Configuration: `configs/data/shopee.yaml`
- OpenCV: `4.12.0`
- Split strategy: `leakage_super_component.v1`, seed `2026`

## Dataset

- Listings: **34,250**
- Label groups: **11,014**
- Unique referenced images: **32,412**
- Missing values / duplicate IDs / decode failures: **0 / 0 / 0**
- Group size median / P95 / max: **2 / 7 / 51**
- Title length median / P95: **53 / 98** characters

![Group sizes](figures/data_audit_v1/group_size_bands.svg)

![Title lengths](figures/data_audit_v1/title_length_histogram.svg)

## Leakage-safe split

- Listings: `{"test": 3429, "train": 27391, "validation": 3430}`
- Label groups: `{"test": 1097, "train": 8817, "validation": 1100}`
- Super-components: **10,866**
- Multi-label components: **129**
- Maximum component: **4 labels / 69 rows**
- Cross-split label groups, exact image references, exact pHashes, and identical image bytes: **0**
- Near-pHash pairs crossing splits: **77**. They are audited but not automatically merged because
  some are valid variants.

![Split counts](figures/data_audit_v1/split_listing_counts.svg)

## Findings

| Code | Severity | Count | Meaning |
|---|---:|---:|---|
| `exact_image_cross_label` | warning | 46 | Exact image reference spans labels |
| `exact_sha_cross_label` | warning | 46 | Exact image bytes span labels |
| `exact_phash_cross_label` | warning | 147 | Exact pHash spans labels |
| `near_phash_cross_label` | warning | 327 | Cross-label pHash pairs have Hamming distance <= 4 |
| `normalized_title_cross_label` | warning | 106 | Normalized title spans labels |

## Manual inspection

A deterministic local gallery covers 24 same-group and 24 difficult cross-group pairs. Spot checks
found both probable fragmented labels and legitimate product variants (for example, 470 ml versus
780 ml packaging). This supports retaining the source labels while treating pHash/title collisions
as audit and hard-negative signals.
