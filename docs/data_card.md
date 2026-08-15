# Data card — Shopee Price Match Guarantee

## Status and provenance

The user-provided authorized Kaggle release was made available locally on 2026-08-16. Its
original download timestamp is not encoded in the files and is therefore unknown.

- Source: Kaggle **Shopee — Price Match Guarantee** competition.
- Access: subject to the competition's account access and rules.
- Redistribution: raw CSVs, images, inspection galleries, and row-level manifests stay local.
- Training metadata SHA-256:
  `82b8ec57e81b603e00c63779b7eeea69a2bd7d9cbb48377657cf547b422f9175`.
- Competition test metadata has 3 rows and 3 available images; its SHA-256 is
  `b881ec236653583e9105fc3a7bec191ac0604852279f42596c12e90c61af1dd6`.
- `sample_submission.csv` is present with SHA-256
  `164bce85bd72fccdac54b3a8842474a90ed5d0b712d36140b437a9c24fa205e2`.
- Phase 1 config: `phase1.data.v1`.
- Pipeline: `phase1.pipeline.v1` using OpenCV `4.12.0`.

## Observed release

`train.csv` has the exact columns `posting_id`, `image`, `image_phash`, `title`, and
`label_group`.

| Property | Observed value |
|---|---:|
| Listings | 34,250 |
| Label groups | 11,014 |
| Referenced/available/decoded images | 32,412 / 32,412 / 32,412 |
| Unique image SHA-256 values | 32,412 |
| Missing values | 0 |
| Duplicate posting IDs / rows | 0 / 0 |
| Group size min / median / P95 / max | 2 / 2 / 7 / 51 |
| Image width and height median | 700 / 700 px |
| Image width/height range | 100–5,000 px |
| Title characters median / P95 / max | 53 / 98 / 357 |
| Title tokens median / P95 / max | 9 / 17 / 61 |

All titles contain Latin characters, none contain non-ASCII characters, and 20,206 titles contain
digits. This release is multilingual at the lexical level but does not provide native-script text
in the observed training titles.

## Label and duplicate audit

- 1,246 filenames are referenced more than once; 46 filename buckets span label groups.
- 3,229 pHash values are repeated; 147 exact-pHash buckets span label groups.
- 327 cross-label pHash pairs have Hamming distance at most 4 (140 at distance 2 and 187 at
  distance 4).
- 106 normalized-title buckets span label groups.
- Exact image bytes do not occur under different filenames, but 46 repeated filenames—and thus
  46 SHA buckets—span labels.

Manual spot checks confirm both kinds of ambiguity. Some cross-label pairs appear to be the same
product or reused listing image, suggesting fragmented labels. Others are legitimate variants:
for example, the same dishwashing-liquid packaging at 470 ml versus 780 ml has the same pHash but
must not be merged blindly. Exact titles can likewise describe visually distinct color variants.
Competition `label_group` is therefore retained as experimental ground truth while ambiguity is
reported explicitly.

## Frozen split

Split strategy `leakage_super_component.v1` (seed 2026) first links label groups connected by the
same filename, image SHA-256, or exact pHash, then assigns the resulting component atomically.

| Split | Listings | Label groups | Super-components |
|---|---:|---:|---:|
| Train | 27,391 | 8,817 | 8,696 |
| Validation | 3,430 | 1,100 | 1,085 |
| Test | 3,429 | 1,097 | 1,085 |

There are 10,866 super-components; 129 connect multiple source labels. The largest contains four
labels and 69 rows. Cross-split leakage counts are zero for label groups, exact filenames, exact
SHA-256 values, and exact pHashes. Seventy-seven near-pHash pairs cross splits; they remain audit
flags instead of hard links because near-pHash alone is not a reliable product-identity rule.

- Local manifest SHA-256:
  `c9cef390b5fbde6c833fddb15a0a8df2c7fbecacd8d50fb83aadba6056bf8e09`.
- The row-level manifest and gallery remain ignored because they expose competition records.
- The test split is frozen and must not be used for model or threshold selection.

## Intended use and limitations

Use this data for controlled exact-product retrieval, pair matching, and catalog entity-resolution
research. Build vocabulary, learned statistics, hard-negative policies, and calibration from
train/validation only.

The labels may contain fragmentation or variant-policy inconsistencies. Product identity depends
on attributes such as size, color, quantity, flavor, and model number that pHash or normalized
titles can obscure. Dataset performance does not establish marketplace, commercial, or legal
identity, and raw competition content must not be published from this repository.
