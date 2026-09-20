# Data Paths Reference

All data required to reproduce evaluation of the six methods (M1–M6) is
bundled inside this repository, with one external dependency: the
full-Africa morphological indicator table, produced by the companion
[`africa_morphology_metrics`](https://github.com/Sean-creater/africa_morphology_metrics)
repository. It is not bundled here and is only needed to apply the trained
M6 student beyond this repository's evaluation set — it is not required to
reproduce `final_ranking.csv`. Point the `MORPHOLOGY_CLEANED_CSV_ROOT`
environment variable at your own checkout of that pipeline's output if it
differs from the documented default.

## Directory Layout

```
morphology_embedding_comparison/
├── data/
│   ├── annotations/    # expert-annotation dataset, train/val split
│   ├── reference/      # evaluation set and final ranking table
│   ├── features/       # morphology feature CSVs
│   └── example/        # illustrative images
├── backbone/dinov3/     # vendored, pruned DINOv3 (weights not bundled)
├── methods/             # one subdirectory per method (M1-M6)
├── shared/               # shared dataset/path utilities
├── eval_all_methods.py   # cross-method evaluation script
└── outputs/              # final checkpoints/embeddings for M1-M6
```

## Python Environment

See `environment.yml` / `requirements.txt`. Training and embedding
extraction require PyTorch + CUDA for the DINOv3-based methods (M3, M4);
the tabular-only methods (M1, M2, M5, M6) do not.
