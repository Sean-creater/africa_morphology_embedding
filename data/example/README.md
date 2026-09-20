# Example / Illustrative Images

This directory bundles a small, reproducible slice of the imagery used in the
study — not the full image sets, which are too large to check into a code
repository. It exists so `methods/m3_frozen_vit/extract.py` and
`methods/m4_finetuned_vit/extract_triplet.py` have a real, local
`VAL_PAIRS_IMAGES_DIR` to run against, and so readers can see what the model
inputs actually look like.

## `val_pair_images/` (30 files, ~104 MB)

Composite validation-pair images (866×1000 px, RGBA), named `A_{GridID}.png` /
`B_{GridID}.png`, matching the naming and lookup convention of
`extract.py` / `extract_triplet.py` (`shared.paths.VAL_PAIRS_IMAGES_DIR`, grid
ID recovered via `stem.lstrip("AB").lstrip("_")`).

`data/reference/eval_pairs.csv` and `data/reference/eval_triplets.csv`
together reference 862 unique grid cells. Of those, only 30 have a
recoverable source image under this lookup convention — the remaining grid
cells' embeddings were extracted at the time of the original study and are
already baked into the ported `.npz` embedding files under `outputs/`
(`shared.paths.FEATURES_VAL_CSV` / `features_val_set.csv` covers the tabular
features for all 862). Re-running `eval_all_methods.py` from the ported
`outputs/` embeddings and reference CSVs reproduces `final_ranking.csv`
without needing any additional images. This bundled subset is provided so the
image-extraction code path (`extract.py` / `extract_triplet.py`) can still be
exercised end-to-end on real data, not as a complete image archive.

Where a grid cell's image existed in more than one source location, the
better-quality "fixed" version was kept (28 of the 30 files here); this
mirrors the override behavior of the original two-tier directory lookup,
now collapsed into the single `VAL_PAIRS_IMAGES_DIR`.

## `annotation_sample/` (8 files, ~4 MB)

A handful of representative 224×224 PNGs drawn from the annotation_set
(714 images / 707 unique grid cells in the original study), one per country,
purely to illustrate the composite satellite + building/road-overlay image
format described in the study. This is not a usable training or evaluation
subset — it is illustrative only.
