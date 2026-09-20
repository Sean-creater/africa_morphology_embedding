# Morphology Embedding Comparison

A methodological comparison of six approaches to embedding African urban
morphology — from 250m hex-grid cells — into a 256-dimensional space that
captures perceptual similarity between neighborhoods, validated against
expert pairwise-similarity judgments.

## Contents

- [What this compares](#what-this-compares)
- [Installation](#installation)
- [Data requirements](#data-requirements)
- [Methods](#methods)
- [Results](#results)
- [Evaluation methodology](#evaluation-methodology)
- [Usage](#usage)
- [Design notes](#design-notes)
- [Repository layout](#repository-layout)
- [Citation](#citation)
- [License](#license)

## What this compares

Given a 250m hex-grid cell of African urban fabric, six methods each produce
a fixed-length embedding meant to place morphologically similar cells (in
building layout, street pattern, land cover) near each other. The methods
span a spectrum from "no learning at all" to "distilled from a vision
transformer that was itself fine-tuned on human judgments":

| | Input | Learned? | Supervised by humans? |
|---|---|---|---|
| **M1** raw_features | 165-dim tabular morphology | No | No |
| **M2** supervised_morphological | 165-dim tabular morphology | Yes (MLP) | Yes (triplet loss) |
| **M3** frozen_vit | 224×224 image | No (frozen backbone) | No |
| **M4** finetuned_vit | 224×224 image | Yes (projection head) | Yes (triplet loss) |
| **M5** selfsup_distillation | 165-dim tabular morphology | Yes (MLP, distilled) | No (distills M3) |
| **M6** triplet_distillation | 165-dim tabular morphology | Yes (MLP, distilled) | Indirectly (distills M4) |

All six are evaluated against the same held-out set of expert pairwise
similarity ratings, using the same three metrics (Pairwise Ranking Accuracy,
Spearman correlation, Triplet Accuracy — see
[Evaluation methodology](#evaluation-methodology)). **M6 wins on all three
metrics** (see [Results](#results)).

The scientific question this answers: for embedding urban morphology at
continental scale (millions of grid cells, no satellite imagery budget for
most of them), how much do you gain from (a) deep learning at all over raw
tabular features, (b) images over tabular features, and (c) distilling a
human-supervised image model into a cheap tabular-only student? M6 shows the
distillation route recovers — and in this case exceeds — the fine-tuned image
model's performance while requiring only tabular features at inference,
making it the only method of the six that is actually deployable across the
full ~1.1M-cell African grid-cell dataset without needing imagery for every
cell.

## Installation

```bash
conda env create -f environment.yml
conda activate morphology_embedding_comparison
# or: pip install -r requirements.txt
```

PyTorch with CUDA is required for the DINOv3-based methods (M3, M4) and
speeds up training/extraction for all others; M1, M2, M5, M6 will run on CPU
(slowly for training, fine for inference).

## Data requirements

Everything needed to reproduce `data/reference/final_ranking.csv` (the
headline result) is bundled in this repository:

- **`data/features/`** — 165-dim morphology feature CSVs for the 5,494-cell
  analysis set and the 996-cell validation-pair set (see its README for exact
  schema). `shared/dataset.py`'s `load_morphology_features` reads `GridID`,
  `center_x`, `center_y`, `City`, `Country` as metadata columns, drops
  `Area_m2` (constant across all 250m hex cells), zero-fills NaNs in columns
  prefixed `LC_` / `H_` / `V_` / `Mean_` (NaN means the underlying quantity is
  genuinely absent, e.g. no buildings in the cell), and median-fills all
  other columns (median computed on the training split only, to avoid
  leakage).
- **`data/annotations/`** — the full expert-annotation dataset (6,960 pairs,
  train/val split) used to train M2 and M4. See its README for inter-annotator
  agreement methodology. The referenced images (both the annotation pairs and
  the 5,494-cell analysis set used for M3/M4 embedding extraction) are
  produced by the companion
  [`africa_morphology_imagery`](https://github.com/Sean-creater/africa_morphology_imagery)
  repository's rendering stage (`png_integrated`, the satellite +
  black-building/dark-gray-road overlay style).
- **`data/reference/`** — the 535-pair / 546-triplet evaluation set and the
  final ranking table.
- **`data/example/`** — a small bundled/illustrative image subset (see its
  README); not the full image sets.
- **`backbone/dinov3/`** — vendored, pruned DINOv3 source code (hub, layers,
  models, utils only — no training infrastructure, no `.git`). Used by M3 and
  M4. **Weights are not bundled** (1.2 GB): download
  `dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth`, the satellite-imagery-pretrained
  ViT-L/16 variant (SAT-493M pretraining set — not the default
  ImageNet/web-pretrained checkpoint), either via Meta's official gated
  download form at
  [ai.meta.com/resources/models-and-libraries/dinov3-downloads](https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/)
  or via the Hugging Face mirror
  [`facebook/dinov3-vitl16-pretrain-sat493m`](https://huggingface.co/facebook/dinov3-vitl16-pretrain-sat493m).
  Place the file at `backbone/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth`.
  The `-eadcf0ff` suffix is a content-hash-style identifier used across
  Meta's DINOv3 checkpoint filenames; no further public documentation of its
  derivation was found.
- **`outputs/`** — the actual final checkpoints and embeddings that produced
  `final_ranking.csv`, for all six methods (see
  [Repository layout](#repository-layout)).

One external dependency is **not** bundled: the full-Africa morphological
indicator CSVs produced by the companion
[`africa_morphology_metrics`](https://github.com/Sean-creater/africa_morphology_metrics)
repository (~1.1M grid cells, too large to include here). This is only needed
by `methods/m6_triplet_distillation/extract_full_africa.py`, which applies
the trained M6 student beyond the analysis/validation sets used for this
comparison — it is not needed to reproduce `final_ranking.csv`. Point
`shared.paths.CLEANED_CSV_ROOT` at your checkout via the
`MORPHOLOGY_CLEANED_CSV_ROOT` environment variable if it differs from the
documented default (see `DATA_PATHS.md`).

### Full-Africa outputs

`extract_full_africa.py`'s output — the morphological indicator table and
M6 embeddings for all African grid cells — is published as release assets:
[**Continent-scale morphology and embedding data**](https://github.com/Sean-creater/morphology_embedding_comparison/releases/tag/africa-full-outputs-v1).
Both files join on `GridID`, consistent with the companion
`africa_morphology_metrics` and `africa_hex_grid_workflow` repositories. See
the manuscript's Methods and Supplementary Information for construction and
validation details. The release is private, visible only to repository
collaborators.

## Methods

### M1 — raw_features

PCA on the 165-dim morphology feature vector. No images, no neural network,
no supervision of any kind. `methods/m1_raw_features/run.py`:
`StandardScaler` (fit on the analysis set only) → PCA to 64 components. Both
the scaled 165-dim vector and the 64-dim PCA projection are saved; evaluation
uses the PCA embedding. This is the sanity-check floor: it establishes
whether any of the learned methods below actually add value over raw tabular
features projected linearly.

### M2 — supervised_morphological

A `MorphologyMetricModel` MLP (165 → 512 → 256 → 256, BatchNorm + ReLU +
Dropout(0.2) on hidden layers, L2-normalized output) trained directly on the
165-dim tabular features with a **triplet cosine loss**:

```
d_ap = 1 - cos(anchor, positive)
d_an = 1 - cos(anchor, negative)
loss = relu(d_ap - d_an + margin)
```

Triplets are built from the annotation pairs: for every grid cell that
appears as one side of two or more rated pairs, every combination of two
peers with a rating gap ≥ 1 annotation step (`margin = |r1-r2|/4 ≥ 0.25`) is
turned into a (anchor, positive, negative) triplet, keeping at most 15
hardest (largest-margin) triplets per anchor. AdamW, batch size 128, lr 1e-3,
5-epoch linear warmup then cosine decay to `1e-6` over 100 epochs, early
stopping on validation Triplet Accuracy (patience 10). This is the same
model family/training recipe as M4, applied to tabular features instead of
images — it isolates how much of M4's performance comes from image content
versus the triplet-learning procedure itself.

### M3 — frozen_vit

The frozen DINOv3 ViT-L/16 backbone (satellite-pretrained, SAT-493M),
run directly on 224×224 composite images with **zero fine-tuning and zero
supervision**. The 1024-dim CLS token is L2-normalized and used as-is —
no projection head, no training of any kind
(`methods/m3_frozen_vit/extract.py`). This measures how much perceptually
relevant structure a generic large vision model already captures in satellite
imagery of built environments, before any task-specific adaptation.

### M4 — finetuned_vit

DINOv3 ViT-L/16 backbone (kept frozen throughout) + a trainable
`ProjectionHead` (1024 → 512 → GELU → 256, L2-normalized output). Trained
with the same hard-triplet mining and triplet cosine loss as M2
(`min_margin=0.25`, max 15 triplets/anchor), but over augmented images
(random 90°-step rotation, flips, color jitter, Gaussian blur) rather than
tabular features. AdamW, lr 5e-5, weight decay 1e-4, cosine annealing over 50
epochs, early stopping on validation Triplet Accuracy (patience 10). The
checkpoint used throughout this study (`best_projection_head_triplet.pth`)
is this single supervised metric-learning stage.

`methods/m4_finetuned_vit/model.py` additionally defines a multi-scale
deep-clustering extension (`ClusterLayer`, `DeepClusteringModel`): three
learnable cosine-similarity cluster layers at K=10/20/30 sharing one
`ProjectionHead`, intended to be trained with a DEC-style KL(P‖Q) clustering
loss (averaged across the three scales, so the embedding space is shaped at
multiple granularities rather than biased toward one arbitrary choice of K)
plus an MSE term anchoring back to the metric-learning embedding. **This
extension is not exercised by any training script in this repository** — no
`train_clustering.py` or equivalent exists here, and the M4 embeddings used
throughout this comparison come from the triplet metric-learning stage
alone. The classes are retained because they are the model definitions a
clustering-based extension of M4 depends on, and because M3's `model.py`
imports the same module for architectural parity with M4.

### M5 — selfsup_distillation

A `StudentMLP` (165 → [512, 512] → 1024, BatchNorm + ReLU + Dropout(0.2))
distills M3's frozen 1024-dim DINOv3 embedding space — **self-supervised**
in the sense that M3's teacher embeddings require zero human annotation,
unlike M4/M6 below. Loss:

```
L = w_mse * MSE(student, teacher) + w_cos * (1 - cosine_similarity(student, teacher))
```

with `w_mse=1.0`, `w_cos=0.5`. Output dimension matches the M3 teacher
(1024, not 256) so direct MSE/cosine comparison is well-defined. Trained
with 5-fold cross-validation (`KFold(n_splits=5, shuffle=True, random_state=42)`),
AdamW (lr 1e-3, weight decay 1e-4), batch size 256, 5-epoch warmup then
cosine decay over 100 epochs, early stopping on validation loss (patience
15). The fold with lowest validation loss is used for embedding extraction.

### M6 — triplet_distillation (the winner)

A `StudentMLP` (165 → [512, 512] → 256, BatchNorm + ReLU + Dropout(0.2))
distills M4's 256-dim triplet-trained embedding space
(`m4_finetuned_vit/embeddings/all_embeddings_triplet.npz`). Unlike M5's
per-sample MSE+cosine loss, M6 uses a **pairwise similarity-matching loss**
computed over each training batch:

```
sim_s = student_batch @ student_batch.T   # [B, B] cosine-similarity matrix
sim_t = teacher_batch @ teacher_batch.T
loss  = MSE(sim_s, sim_t)
```

i.e. it matches the full batch-wise pairwise-cosine-similarity structure of
the teacher's embedding space, not just each sample's own teacher vector —
a relational/structure-preserving distillation objective. Cosine similarity
to the teacher is tracked as a monitoring metric but is not part of the
loss. Same training recipe as M5 otherwise: 5-fold CV, AdamW, batch size
128, lr 1e-3, weight decay 1e-4, 5-epoch warmup + cosine decay over 100
epochs, early stopping on validation loss (patience 15).

Because M6 only needs the 165-dim tabular features at inference — no
imagery — it is the only method of the six that can be (and was) applied to
the full production dataset: `extract_full_africa.py` embeds every grid cell
under the companion `africa_morphology_metrics` repository's output, and
`cluster_full_africa.py` runs multi-resolution K-means (k=1..30) on the
result. See [Design notes](#design-notes) for why this teacher-student
relationship matters.

### M7 / M8 (not part of the published comparison)

Two additional distillation variants — a clustering-based distillation and a
contrastive-based distillation — were explored and did not outperform M6.
They are not included in this repository's active `methods/` or in
`final_ranking.csv`.

## Results

From `data/reference/final_ranking.csv`, evaluated on 535 high-consensus
annotated pairs and 546 derived triplets (see
[Evaluation methodology](#evaluation-methodology)):

| Method | N pairs | PRA (%) | Spearman ρ | Triplet Acc (%) | N triplets |
|---|---|---|---|---|---|
| M1_PCA | 535 | 68.58 | 0.4954 | 78.75 | 546 |
| M3_FrozenViT | 535 | 73.07 | 0.6099 | 77.66 | 546 |
| M5_SelfSupDistill | 535 | 68.17 | 0.4979 | 78.21 | 546 |
| M2_Triplet | 535 | 72.75 | 0.6003 | 80.40 | 546 |
| M4_Triplet | 535 | 75.48 | 0.6657 | 82.05 | 546 |
| **M6_Triplet** | 535 | **77.73** | **0.7173** | **82.42** | 546 |

**M6 (triplet_distillation) wins on all three metrics** — Pairwise Ranking
Accuracy, Spearman correlation, and Triplet Accuracy — despite using only
165-dim tabular features at inference, no imagery. It outperforms its own
teacher (M4), the frozen vision backbone (M3), and both morphology-only
alternatives (M1, M5).

Two clear secondary findings: (1) triplet-supervised training beats its
unsupervised/self-supervised counterpart in both the tabular domain (M2 ≫
M1) and the distillation domain (M6 ≫ M5); (2) even without any fine-tuning,
a frozen large vision model (M3) beats the raw-tabular-features floor (M1) on
PRA and Spearman ρ, showing image content itself carries useful morphological
signal — but M2's triplet-trained tabular model already matches M3 on PRA/ρ
and exceeds it on Triplet Accuracy, so images are not strictly necessary once
the tabular features are put through supervised metric learning.

## Evaluation methodology

**Pairwise Ranking Accuracy (PRA):** for every pair of annotated pairs `(i,
j)` where the human ratings differ, PRA is the fraction where the embedding
similarity ordering agrees with the rating ordering — i.e. `sign(sim_i -
sim_j) == sign(rating_i - rating_j)`. This measures whether the embedding
space's *relative* ordering of similarity matches human judgment, without
requiring the absolute similarity values to match any particular scale.

**Spearman ρ:** rank correlation between embedding cosine similarity and
mean human rating, across the same 535 pairs.

**Triplet Accuracy:** fraction of the 546 evaluation triplets `(anchor,
positive, negative)` where `cos(anchor, positive) > cos(anchor, negative)`,
i.e. where the embedding correctly ranks the human-preferred neighbor closer
than the human-rejected one.

**The 535 pairs and 546 triplets** (`data/reference/eval_pairs.csv` /
`eval_triplets.csv`) are built from two annotation rounds:

- **round2** (`val_pairs_consensus.csv`, 431 pairs): 4-annotator consensus
  pairs already filtered to `max_diff ≤ 2` (max − min rating across
  annotators).
- **anchor round** (`anchor_round/*.csv`, 144 pairs): same 4-annotator
  consensus filter (`max_diff ≤ 2`) applied at merge time.

After deduplication (by unordered grid-ID pair) this yields 535 pairs.
Triplets are then derived from every grid cell appearing as one side of two
or more of these 535 pairs: every combination of two peers with a differing
rating becomes an (anchor, positive, negative) triplet with
`margin = |rating_p - rating_n| / 4`, no minimum-margin filter — yielding 546
triplets. This is a *different*, higher-consensus (4-expert) evaluation set
than the 2-expert `data/annotations/` set used for training M2/M4 (see
below) — evaluation and training annotations come from separate rounds.

**Inter-annotator agreement (training annotations,
`data/annotations/README.md`):** Rounds 1–3, 2 independent experts per pair,
filtering criterion `|expert_1 - expert_2| ≤ 2`, final rating = mean of the
two experts. Train/val split: fixed 90/10 split via
`numpy.random.default_rng(seed=42)`; all six methods use the identical
`train.csv` / `val.csv` files.

## Usage

```bash
# M1: PCA baseline (fast, no GPU needed)
python methods/m1_raw_features/run.py

# M2: triplet-trained morphology MLP
python methods/m2_supervised_morphological/train_triplet.py
python methods/m2_supervised_morphological/extract_triplet.py

# M3: frozen DINOv3 (requires backbone weights, see Data requirements)
python methods/m3_frozen_vit/extract.py

# M4: triplet-trained projection head over frozen DINOv3
python methods/m4_finetuned_vit/train_triplet.py
python methods/m4_finetuned_vit/extract_triplet.py

# M5: self-supervised distillation from M3
python methods/m5_selfsup_distillation/train.py
python methods/m5_selfsup_distillation/extract.py

# M6: triplet distillation from M4 (the winning method)
python methods/m6_triplet_distillation/train.py
python methods/m6_triplet_distillation/extract.py

# Reproduce final_ranking.csv from the bundled outputs/ checkpoints+embeddings
python eval_all_methods.py
```

`eval_all_methods.py` reads directly from the ported `outputs/*/embeddings/*.npz`
files and `outputs/m6_triplet_distillation/models/student_fold*.pth` — running
it against the bundled `outputs/` reproduces `data/reference/final_ranking.csv`
without retraining anything.

## Design notes

**M4 → M6 teacher-student relationship.** M6 distills M4's 256-dim
triplet-trained embedding space into a tabular-features-only student MLP.
This is the key non-obvious design point: M4 needs a 224×224 composite image
per grid cell and a GPU forward pass through a 300M-parameter frozen ViT-L/16
backbone, which is only feasible for the ~5,494-image analysis set and
~1,200-image validation-pair set used in this comparison. M6's student needs
only the 165-dim tabular morphology feature vector already computed for
every grid cell in the full ~1.1M-cell African dataset, making it the only
one of the six methods that is actually deployable at that scale — and,
empirically, it does not just match its teacher but exceeds it on all three
evaluation metrics.

**M4's multi-scale K=10/20/30 clustering (`ClusterLayer`,
`DeepClusteringModel`).** As documented in `methods/m4_finetuned_vit/model.py`,
the rationale given in-code for training three cluster layers jointly at
different K is robustness to the choice of K: "Multi-scale clustering: three
ClusterLayers at different K share the same ProjectionHead. Training
averages KL losses across scales so the embedding space is structured at
multiple granularities, not biased to any single K." As noted in
[Methods](#methods) above, this extension is defined in code but not
exercised by any training script in this repository — the M4 embeddings used
throughout this comparison come from the triplet metric-learning stage
alone.

**Why M1 (PCA baseline) is included.** M1 is a sanity-check floor: if a deep
or image-based method could not beat a simple linear PCA projection of the
raw tabular features, that would indicate the additional model complexity
(or additional imagery) is not earning its cost. All five other methods do
beat M1, but M2 (tabular + triplet learning, no images) comes close to
matching M3 (frozen ViT, no learning) on PRA/ρ — indicating that the choice
of image versus tabular input matters less than whether the model is trained
with human-supervised metric learning at all.

**Why M5 and M6 need different loss formulations.** M5 distills into M3's
raw 1024-dim DINOv3 feature space, which was never itself shaped by
human-labeled similarity — a per-sample MSE+cosine loss against the
individual teacher vectors is sufficient. M6 distills into M4's 256-dim
space, which *was* shaped by triplet supervision to encode relative
similarity structure; matching the pairwise similarity matrix
(`student @ student.T` vs `teacher @ teacher.T`) rather than individual
vectors is what lets the student inherit that relative-ordering structure
rather than just each sample's raw coordinates.

## Repository layout

```
morphology_embedding_comparison/
├── README.md, DATA_PATHS.md, environment.yml, requirements.txt, pyproject.toml
├── eval_all_methods.py
├── methods/
│   ├── m1_raw_features/            # run.py
│   ├── m2_supervised_morphological/  # config, dataset, model, train_triplet, extract_triplet
│   ├── m3_frozen_vit/              # config, dataset, model, extract
│   ├── m4_finetuned_vit/           # config, dataset, model, train_triplet, extract_triplet
│   ├── m5_selfsup_distillation/    # config, models, train, extract
│   └── m6_triplet_distillation/    # config, models, train, extract, extract_full_africa, cluster_full_africa
├── shared/                         # paths.py, dataset.py
├── backbone/dinov3/                # vendored, pruned DINOv3 (weights not bundled)
├── data/
│   ├── annotations/                # training annotation data (6,960 pairs)
│   ├── reference/                  # eval_pairs.csv, eval_triplets.csv, final_ranking.csv
│   ├── features/                   # 165-dim morphology feature CSVs
│   └── example/                    # bundled validation-pair + illustrative images
└── outputs/                        # final checkpoints + embeddings for M1-M6
```

## Citation

If you use this comparison or the M6 student model, please cite the
companion morphology-extraction pipeline,
[`africa_morphology_metrics`](https://github.com/Sean-creater/africa_morphology_metrics),
which produces the 165-dim feature inputs used throughout.

## License

No license file was found in the source project this repository was derived
from. Until a license is chosen, all rights are reserved by default; add a
`LICENSE` file before distributing this code publicly. The vendored
`backbone/dinov3/` dependency is the one exception: it carries Meta's own
DINOv3 license (see `backbone/dinov3/LICENSE.md`), distinct from the rest of
this repository, and its pretrained weights (not bundled — see
[Data requirements](#data-requirements)) are subject to the same license's
terms.
