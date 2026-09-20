# Annotation Data

## Files

| File | Description | Pairs |
|------|-------------|-------|
| `pairs_all_rounds.csv` | Full consolidated annotation dataset (Rounds 1–3, inter-annotator filtered, averaged) | 6,960 |
| `pairs_all_rounds_detailed.csv` | Same as above but with per-expert ratings retained | 6,960 |
| `train.csv` | Training split (90%, random seed 42) | 6,264 |
| `val.csv` | Validation split (10%, random seed 42) | 696 |

## Format

```
Image_A_gridID, Image_B_gridID, Image_A, Image_B, Rating
```

- `Image_A_gridID`, `Image_B_gridID`: Grid cell identifiers (e.g. `EGY_0110_060235`)
- `Image_A`, `Image_B`: Image filenames. In the original study these were found in
  `data/images/annotation_set/` (714 images, 707 unique grid cells); this repo
  bundles only a small illustrative sample under `data/example/` (see its README).
- `Rating`: Mean expert similarity score (1.0–5.0, step 0.5)
  - 1.0 = completely dissimilar morphologies
  - 5.0 = very similar morphologies

## Inter-annotator Agreement

- Rounds 1 & 2: 2 independent experts per pair
- Round 3: 2 independent experts per pair
- Filtering criterion: |expert_1 − expert_2| ≤ 2
- Final rating: mean of two experts

## Train/Val Split

Fixed split generated with `numpy.random.default_rng(seed=42)`, stratified by random permutation.
All six methods use the identical `train.csv` / `val.csv` files.
