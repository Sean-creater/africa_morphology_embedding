# Data Paths Reference

All data required to reproduce evaluation of the six methods (M1–M6) is
bundled inside this repository, with one external dependency: the full-Africa
morphological indicator CSVs (see below), which are the output of a separate,
already-published companion repository.

## Directory Layout

```
morphology_embedding_comparison/
├── data/
│   ├── annotations/
│   │   ├── pairs_all_rounds.csv          # 6,960 annotated pairs (Rounds 1-3)
│   │   ├── pairs_all_rounds_detailed.csv
│   │   ├── train.csv                     # 6,264 pairs (90%, seed=42)
│   │   ├── val.csv                       # 696 pairs  (10%, seed=42)
│   │   └── README.md
│   ├── reference/
│   │   ├── eval_pairs.csv                # 535 high-consensus pairs (round2 + anchor)
│   │   ├── eval_triplets.csv             # 546 triplets derived from eval_pairs
│   │   └── final_ranking.csv             # headline M1-M6 comparison result
│   ├── features/
│   │   ├── features_analysis_set.csv     # 165-dim features, 5,494 grid cells
│   │   ├── features_val_set.csv          # 165-dim features, 996 grid cells
│   │   ├── col_name_mapping.csv
│   │   └── README.md
│   └── example/
│       ├── val_pair_images/              # 30 bundled validation-pair images
│       ├── annotation_sample/            # 8 illustrative annotation_set images
│       └── README.md
│
├── backbone/
│   └── dinov3/                           # vendored, pruned DINOv3 source (no weights)
│       └── dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth   # NOT bundled — see README
│
├── methods/
│   ├── m1_raw_features/
│   ├── m2_supervised_morphological/
│   ├── m3_frozen_vit/
│   ├── m4_finetuned_vit/
│   ├── m5_selfsup_distillation/
│   └── m6_triplet_distillation/
│
├── shared/                               # shared utilities (dataset loader, path constants)
├── eval_all_methods.py                   # cross-method evaluation script
└── outputs/                              # ported final checkpoints/embeddings (6 methods only)
```

## Morphological Features (external, companion repository)

The 165-dim morphological indicator CSVs for the full set of African grid
cells are produced by the companion
[`africa_morphology_metrics`](https://github.com/Sean-creater/africa_morphology_metrics)
repository and are **not** bundled here (they cover ~1.1M grid cells across
8,393 cities — far too large for this repo). `shared/paths.py` resolves them
via:

```python
CLEANED_CSV_ROOT = Path(os.environ.get(
    "MORPHOLOGY_CLEANED_CSV_ROOT",
    "/home/sjs96_file_share/Momepy_workflow/Metrix/data_cleaning_20260224/Cleaned",
))
```

Set the `MORPHOLOGY_CLEANED_CSV_ROOT` environment variable to point at your own
checkout of that pipeline's cleaned-CSV output
(`{Country}/{City}_grid_250m_hex_enriched_metrics_v15.csv`) if it differs from
the documented default. This path is only used by
`methods/m6_triplet_distillation/extract_full_africa.py`, which applies the
trained M6 student to the full production dataset — it is not required to
reproduce `final_ranking.csv`, which uses the bundled `data/features/*.csv`
and `outputs/*/embeddings/*.npz` files directly.

## Python Environment

See `environment.yml` / `requirements.txt`. Training and embedding extraction
require PyTorch + CUDA; morphology-only methods (M1, M2, M5, M6) do not
require the DINOv3 backbone or GPU image processing.
