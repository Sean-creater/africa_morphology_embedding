# Morphological Features

## Files

| File | Description |
|------|-------------|
| `features_analysis_set.csv` | 165-dim morphological indicators for all 5,494 grid cells in the analysis set |
| `features_val_set.csv` | 165-dim morphological indicators for the 996 grid cells used to build the validation-pair evaluation set |
| `col_name_mapping.csv` | Full name mapping for truncated column names |

## Format

- **Rows:** 5,494 (`features_analysis_set.csv`) / 996 (`features_val_set.csv`) — one per grid cell
- **Metadata columns:** `GridID`, `center_x`, `center_y`, `City`, `Country`
- **Excluded column:** `Area_m2` (constant across all 250m hex cells, dropped by `shared.dataset.load_morphology_features`)
- **Feature columns (165):** land cover (`LC_*`), building height (`H_*`), building volume (`V_*`), street-network topology, and building/plot morphology indicators
- **GridID** links `features_analysis_set.csv` to `data/images/analysis_set/{City}/{GridID}_{variant}.png` (illustrative sample only — see `data/example/`)

## Coverage

46 African cities across 20+ countries. All GridIDs are verified to exist in both the
image dataset and the full-Africa morphological indicator database produced by the
companion [`africa_morphology_metrics`](https://github.com/Sean-creater/africa_morphology_metrics)
repository (see `shared/paths.py`'s `CLEANED_CSV_ROOT`).

## NaN Handling

Features with `LC_` / `H_` / `V_` / `Mean_` prefixes: fill with 0 (NaN means the
underlying quantity is genuinely absent — e.g. no buildings in the cell — not missing).
All other features: fill with the column median, computed on the training split only
to avoid leakage (`shared.dataset.load_morphology_features`).
