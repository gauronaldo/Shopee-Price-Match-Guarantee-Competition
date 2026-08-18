# Shopee competition solution and repository review

Reviewed on 2026-08-18. This is a design review, not a reproduction claim. Public write-ups are
often leaderboard-oriented, may depend on pretrained/external artifacts, and do not necessarily
document leakage-safe offline validation to this repository's standard.

## Executive summary

Strong Shopee solutions converged on a multi-stage retrieval system rather than a single binary
classifier:

1. train or fine-tune separate image and text embedding models;
2. retrieve a high-recall multimodal candidate set;
3. rescore pairs using similarities and neighborhood/graph context;
4. use query expansion, graph reasoning, clustering, voting, or conservative post-processing;
5. tune thresholds for mean per-query F1 under strict runtime and memory constraints.

This broadly validates the repository roadmap. The ideas should be adopted in phase order, but
the pretrained backbones, leaderboard threshold hacks, and large ensembles must not enter the
scratch track.

## What leading solutions did

### First place

The first-place public description used NFNet for images and several BERT/XLM-family text
encoders, trained with ArcFace-style metric learning. It combined image-only, text-only, and
concatenated multimodal neighborhoods, then used Iterative Neighborhood Blending (INB): update a
query embedding from sufficiently similar neighbors and rerun nearest-neighbor search. Public
summaries also describe scheduled/increasing ArcFace margins, warmup, gradient clipping, and
class-size-adaptive margins. Joint image-text training was reportedly weaker than strong separate
encoders plus inference-time combination.

Sources: [first-place discussion](https://www.kaggle.com/c/shopee-product-matching/discussion/238136),
[winner's portfolio summary](https://portfoly-yoonniverse.vercel.app/user/62d53db4ca5fd788ffaa8727/yoonsoo-kim),
and the [top-solution comparison](https://aimaster.tistory.com/98).

### Second place

The second-place team used a two-stage system. Stage 1 produced image, text, and multimodal
similarities using metric-learned NFNet/DeiT image embeddings, train-time TF-IDF, and pretrained
Indonesian/multilingual BERT or paraphrase-XLM text embeddings. CurricularFace worked better than
the alternatives they tried for image metric learning.

Stage 2 predicted whether a candidate pair matched. Their LightGBM features included image/text
similarities, edit distance, neighborhood similarity statistics, image dimensions/file size, and
title length/token counts. A later Graph Attention Network treated listings as nodes and
candidate relations as edges, using only four edge signals: image, BERT, multimodal, and TF-IDF
similarities. They also used query expansion and graph features. The public slide deck records a
progression from a simple NFNet+TF-IDF threshold system in the 0.74 range to a final 0.792 public
score, with pair modeling, stronger text encoders, query expansion, and GAT contributing in
stages.

Operational engineering mattered: GPU FAISS, GPU forest inference, faster image loading, and
process isolation were used to stay within the Kaggle notebook time/memory budget.

Sources: [second-place deck](https://speakerdeck.com/lyakaap/shopee-2nd-place-solutiontoshang-wei-jie-fa-matome),
[second-place discussion](https://www.kaggle.com/c/shopee-product-matching/discussion/238022),
and [DeNA's result announcement](https://dena.com/jp/news/3675a/).

### Other high-ranking patterns

The second-place authors' comparison summarizes the next systems as follows:

| Rank | Retrieval representations | Decision/grouping layer |
|---:|---|---|
| 3rd | EfficientNet, DINO/ViT, CLIP, BERT, TF-IDF | CatBoost plus agglomerative clustering |
| 4th | NFNet/EfficientNet, BERT, TF-IDF | Weighted similarities plus query expansion |
| 5th | NFNet/EfficientNet/ViT/Swin/ResNet, BERT/TF-IDF/MLP | XGBoost, DBA, agglomerative clustering |
| 6th | Joint multi-task image, text, and multimodal embeddings | Voting over cosine and Euclidean neighborhoods |

The sixth-place summary also describes extracting quantities/units and rejecting matches when
those attributes conflict. This is directly relevant to exact-product identity, where 200 g and
500 g must not be merged merely because brand and packaging are similar.

Source: [top-solution comparison](https://aimaster.tistory.com/98) and the
[second-place deck](https://speakerdeck.com/lyakaap/shopee-2nd-place-solutiontoshang-wei-jie-fa-matome).

### A documented silver-medal pipeline

A 44th-place retrospective used CNN+ArcFace image embeddings, TF-IDF and BERT/Indonesian-BERT
text embeddings, KNN retrieval, concatenated embeddings, model voting, and a fallback that
relaxed recall when a query matched only itself. The author reports that voting and this fallback
helped, while TTA did not help in their setup.

Source: [silver solution retrospective](https://developer.volcengine.com/articles/7387237101139410963).

## Repository review

### `jingxuanyang/Shopee-Product-Matching`

This repository separates image notebooks, text notebooks, image-text ensemble notebooks,
post-processing, tests, and a written report. It demonstrates the useful conceptual separation
between modality training and fusion. Its documented best text path uses a pretrained
multilingual sentence-transformer checkpoint and searches an F1 threshold.

However, its setup instructions download a dataset mirror and pretrained weights from external
storage. That is incompatible with this project's rule that the user obtains the competition
data through their own Kaggle account and that raw data is never redistributed. It is a useful
algorithm reference, not a reproducibility template for this repository.

Source: [`jingxuanyang/Shopee-Product-Matching`](https://github.com/jingxuanyang/Shopee-Product-Matching).

### `Jinal17/Ecommerce-Product-Matching`

This repository exposes separate NFNet, EfficientNet, ResNet, XLM-R text, custom multimodal, and
ensemble scripts. Its inference path uses GPU KNN over image/text embeddings and unions those
predictions with pHash matches. This is representative of many competition pipelines.

The code is still close to a Kaggle notebook export: paths and GPU assumptions are embedded in
scripts, thresholds are literal values, and the README does not document a group-disjoint local
validation protocol, test suite, or configuration/provenance system comparable to this project.
The architecture ideas are useful; the engineering structure should not be copied.

Source: [`Jinal17/Ecommerce-Product-Matching`](https://github.com/Jinal17/Ecommerce-Product-Matching)
and its [inference script](https://github.com/Jinal17/Ecommerce-Product-Matching/blob/main/shopee_inference.py).

## What this project should adopt

| Competition lesson | Project phase | Controlled adaptation |
|---|---:|---|
| Strong separate image/text embeddings | 3–5 | Train both scratch encoders independently before learned fusion |
| Product-aware metric learning | 3–5 | Start with supervised contrastive/batch-hard loss; ablate ArcFace-like objectives later |
| High-recall candidate union | 7 | Union label-blind image/text neighbors; report candidate Recall@K separately |
| Pair-level meta-model | 5 and 8 | Use symmetric pair features and calibration, not an opaque leaderboard-only stack |
| Quantity/model/unit conflicts | 4, 6, 8 | Preserve digits/units, mine variant hard negatives, add conservative consistency features |
| Query expansion/neighborhood context | 7–8 | Evaluate only after exact retrieval is verified; guard against transitive false merges |
| Graph link prediction/clustering | 8 | Begin with reciprocal-kNN and conservative components before GNN complexity |
| Efficiency engineering | 7 and 10 | Measure exact-search cost first, then FAISS and optimized extraction with parity tests |

Phase 2 already confirms one competition-wide observation: character TF-IDF is a very strong
baseline, while image evidence helps when fused carefully. The next honest experiment remains a
small scratch image encoder, not a pretrained NFNet/BERT ensemble.

## What this project should not copy

- leaderboard threshold tuning or assumptions about hidden test size/distribution;
- forcing every query to have at least two matches solely because it improves competition F1;
- pretrained image/text models before the scratch track is complete;
- raw-data mirrors, bundled checkpoints, or undocumented external artifacts;
- monolithic inference notebooks with hard-coded paths and thresholds;
- large ensembles before single-model failure modes and compute costs are understood;
- connected-component expansion without measuring false merges.

The competition solutions optimize a constrained Kaggle submission. This repository instead
targets a defensible catalog entity-resolution system, so validation discipline, pair
calibration, cluster safety, reproducibility, and evidence for manual review take priority over
leaderboard-specific post-processing.

## Sources reviewed

- [Shopee competition page](https://www.kaggle.com/competitions/shopee-product-matching)
- [First-place discussion](https://www.kaggle.com/c/shopee-product-matching/discussion/238136)
- [Second-place discussion](https://www.kaggle.com/c/shopee-product-matching/discussion/238022)
- [Second-place technical deck](https://speakerdeck.com/lyakaap/shopee-2nd-place-solutiontoshang-wei-jie-fa-matome)
- [DeNA announcement for 2nd and 10th place](https://dena.com/jp/news/3675a/)
- [Top-solution comparison](https://aimaster.tistory.com/98)
- [Silver solution retrospective](https://developer.volcengine.com/articles/7387237101139410963)
- [`jingxuanyang/Shopee-Product-Matching`](https://github.com/jingxuanyang/Shopee-Product-Matching)
- [`Jinal17/Ecommerce-Product-Matching`](https://github.com/Jinal17/Ecommerce-Product-Matching)
- [Public notebook index for the competition](https://readmedium.com/major-compilation-best-notebooks-on-kaggle-part-1-12ebd4c1a27a)
