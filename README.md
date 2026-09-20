# Morphology Embedding Comparison

A methodological comparison of six approaches to embedding African urban
morphology — from 250m hex-grid cells — into a compact space that captures
perceptual similarity between neighborhoods, validated against expert
pairwise-similarity judgments.

## What this compares

Six methods each produce an embedding meant to place morphologically
similar grid cells near each other, spanning a spectrum from unsupervised
linear projection to a vision-transformer embedding distilled into a
tabular-features-only student:

| | Input | Learned? | Supervised by humans? |
|---|---|---|---|
| **M1** raw_features | tabular morphology | No | No |
| **M2** supervised_morphological | tabular morphology | Yes (MLP) | Yes |
| **M3** frozen_vit | image | No (frozen backbone) | No |
| **M4** finetuned_vit | image | Yes (projection head) | Yes |
| **M5** selfsup_distillation | tabular morphology | Yes (distilled) | No (distills M3) |
| **M6** triplet_distillation | tabular morphology | Yes (distilled) | Indirectly (distills M4) |

All six are evaluated against the same held-out expert pairwise-similarity
ratings (see [Evaluation methodology](#evaluation-methodology)). M6 wins on
all three evaluation metrics (see [Results](#results)).

The question this addresses: for embedding urban morphology at continental
scale, where satellite imagery is not available or practical for every
cell, how much is gained from deep learning over raw tabular features, from
images over tabular features, and from distilling a human-supervised image
model into a tabular-only student? M6 shows the distillation route matches
— and in this case exceeds — its teacher's performance while requiring only
tabular features at inference, making it the only method of the six
deployable across the full African grid-cell dataset.

## Installation

```bash
conda env create -f environment.yml
conda activate morphology_embedding_comparison
# or: pip install -r requirements.txt
```

PyTorch with CUDA is required for the DINOv3-based methods (M3, M4).

## Data requirements

Everything needed to reproduce `data/reference/final_ranking.csv` is
bundled in this repository: morphology feature CSVs (`data/features/`),
the expert-annotation dataset (`data/annotations/`), the evaluation set and
final ranking table (`data/reference/`), a small illustrative image subset
(`data/example/`), the trained checkpoints and embeddings for all six
methods (`outputs/`), and a vendored, pruned copy of DINOv3
(`backbone/dinov3/`; pretrained weights are not bundled — see
`DATA_PATHS.md` for how to obtain them).

Images are produced by the companion
[`africa_morphology_imagery`](https://github.com/Sean-creater/africa_morphology_imagery)
repository. The full-Africa morphological indicator table used to apply M6
beyond this comparison's evaluation set is produced by the companion
[`africa_morphology_metrics`](https://github.com/Sean-creater/africa_morphology_metrics)
repository and is not bundled here; see `DATA_PATHS.md`.

### Full-Africa outputs

`extract_full_africa.py`'s output — the morphological indicator table and
M6 embeddings for all African grid cells — is published as release assets:
[**Continent-scale morphology and embedding data**](https://github.com/Sean-creater/morphology_embedding_comparison/releases/tag/africa-full-outputs-v1).
Both files join on `GridID`, consistent with the companion repositories.
See the manuscript's Methods and Supplementary Information for construction
and validation details. The release is private, visible only to repository
collaborators.

## Methods

- **M1 — raw_features**: PCA on the tabular morphology feature vector. No
  learning, no supervision. The sanity-check floor.
- **M2 — supervised_morphological**: an MLP trained directly on tabular
  features with a triplet similarity loss derived from the annotation
  data. Isolates how much of M4's performance comes from image content
  versus the training procedure itself.
- **M3 — frozen_vit**: the frozen DINOv3 backbone run directly on composite
  images, with no fine-tuning or supervision. Measures how much
  perceptually relevant structure a generic vision model already captures.
- **M4 — finetuned_vit**: the frozen DINOv3 backbone plus a trainable
  projection head, trained with the same triplet objective as M2 but over
  images. This is the "teacher" model M6 distills from.
- **M5 — selfsup_distillation**: an MLP distilling M3's frozen embedding
  space into tabular features — self-supervised, since M3's teacher
  embeddings require no human annotation.
- **M6 — triplet_distillation (the winner)**: an MLP distilling M4's
  triplet-trained embedding space into tabular features, using a
  relational (pairwise-similarity-matching) objective rather than
  per-sample matching. Because it needs only tabular features at
  inference, it is the only method of the six applied to the full
  production dataset.
- **M7 / M8** (not part of the published comparison): two additional
  distillation variants were explored and did not outperform M6.

See the manuscript's Methods and Supplementary Information for full
architecture, training, and loss-function details for each method.

## Results

From `data/reference/final_ranking.csv`, evaluated against a held-out set
of expert-annotated pairs and derived triplets (see
[Evaluation methodology](#evaluation-methodology)):

| Method | PRA (%) | Spearman ρ | Triplet Acc (%) |
|---|---|---|---|
| M1_PCA | 68.58 | 0.4954 | 78.75 |
| M3_FrozenViT | 73.07 | 0.6099 | 77.66 |
| M5_SelfSupDistill | 68.17 | 0.4979 | 78.21 |
| M2_Triplet | 72.75 | 0.6003 | 80.40 |
| M4_Triplet | 75.48 | 0.6657 | 82.05 |
| **M6_Triplet** | **77.73** | **0.7173** | **82.42** |

M6 wins on all three metrics despite using only tabular features at
inference, outperforming its own teacher (M4), the frozen vision backbone
(M3), and both morphology-only alternatives (M1, M5). See the manuscript
for full discussion.

## Evaluation methodology

Three metrics, computed against a held-out, high-consensus expert-annotated
evaluation set distinct from the training annotation data:

- **Pairwise Ranking Accuracy (PRA)**: agreement between embedding
  similarity ordering and human rating ordering, across all comparable
  pairs of annotated pairs.
- **Spearman ρ**: rank correlation between embedding similarity and mean
  human rating.
- **Triplet Accuracy**: fraction of evaluation triplets where the
  embedding correctly ranks the human-preferred neighbor closer.

See the manuscript's Methods and Supplementary Information for the exact
evaluation-set construction and inter-annotator agreement methodology.

## Usage

```bash
# M1: PCA baseline
python methods/m1_raw_features/run.py

# M2: triplet-trained morphology MLP
python methods/m2_supervised_morphological/train_triplet.py
python methods/m2_supervised_morphological/extract_triplet.py

# M3: frozen DINOv3
python methods/m3_frozen_vit/extract.py

# M4: triplet-trained projection head over frozen DINOv3
python methods/m4_finetuned_vit/train_triplet.py
python methods/m4_finetuned_vit/extract_triplet.py

# M5: self-supervised distillation from M3
python methods/m5_selfsup_distillation/train.py
python methods/m5_selfsup_distillation/extract.py

# M6: triplet distillation from M4
python methods/m6_triplet_distillation/train.py
python methods/m6_triplet_distillation/extract.py

# Reproduce final_ranking.csv from the bundled outputs/
python eval_all_methods.py
```

## Repository layout

```
morphology_embedding_comparison/
├── README.md, DATA_PATHS.md, environment.yml, requirements.txt, pyproject.toml
├── eval_all_methods.py
├── methods/                         # one subdirectory per method (M1-M6)
├── shared/                          # shared path/dataset utilities
├── backbone/dinov3/                 # vendored, pruned DINOv3 (weights not bundled)
├── data/
│   ├── annotations/                  # training annotation data
│   ├── reference/                    # evaluation set + final ranking table
│   ├── features/                     # morphology feature CSVs
│   └── example/                      # illustrative images
└── outputs/                          # final checkpoints + embeddings for M1-M6
```

## Citation

If you use this comparison or the M6 student model, please cite the
companion morphology-extraction pipeline,
[`africa_morphology_metrics`](https://github.com/Sean-creater/africa_morphology_metrics),
which produces the feature inputs used throughout.

## License

No license file was found in the source project this repository was derived
from. Until a license is chosen, all rights are reserved by default; add a
`LICENSE` file before distributing this code publicly. The vendored
`backbone/dinov3/` dependency is the one exception: it carries Meta's own
DINOv3 license (see `backbone/dinov3/LICENSE.md`), and its pretrained
weights (not bundled) are subject to the same license's terms.
