# Africa Morphology Embedding

Produces a compact, continent-scale embedding of African urban morphology,
capturing perceptual similarity between neighborhoods, for downstream use
in constructing a data-driven typology of African urban form.

## Motivation

Two common approaches to representing urban morphology each have
limitations. Indicator-based approaches (morphology metrics computed
directly from building footprints and street networks) are interpretable
and lightweight, but have limited representational capacity and are often
built around indicators derived from non-African urban contexts. Visual
embedding approaches (learned directly from imagery) have strong
representational power and can incorporate visual priors and expert
judgment, but are difficult to interpret and computationally expensive at
continental scale.

This repository combines both: a large vision model, guided by expert
similarity judgments, learns a strong embedding space from urban imagery;
a lightweight model then reproduces that embedding space from
interpretable morphology indicators alone. The result keeps the
representational strength and expert grounding of the image-based
approach, while remaining interpretable and cheap enough to run across
every urban grid cell on the continent. The resulting embeddings exhibit
clear clustering structure and are used downstream to construct an urban
form typology (a separate, companion analysis).

## Approach

Urban imagery, guided by expert similarity judgments, trains an embedding
space (the "teacher" stage) that captures how African urban forms differ
from one another. A lightweight model then learns to reproduce that
embedding space from morphology indicators alone (the "student" stage),
so the final embedding requires no imagery at inference and can be
computed for every grid cell on the continent.

To validate this design, six alternative ways of producing the embedding
were compared against the same held-out expert similarity judgments — see
[Design validation](#design-validation). The teacher-student approach
described above performed best on every evaluation metric and is the one
used to produce the published embeddings.

## Installation

```bash
conda env create -f environment.yml
conda activate africa_morphology_embedding
# or: pip install -r requirements.txt
```

PyTorch with CUDA is required for the image-based stages.

## Data requirements

Everything needed to reproduce the headline result is bundled in this
repository: morphology feature CSVs (`data/features/`), the
expert-annotation dataset (`data/annotations/`), the evaluation set and
ranking table (`data/reference/`), a small illustrative image subset
(`data/example/`), the trained checkpoints and embeddings, and a vendored,
pruned copy of the vision backbone (`backbone/dinov3/`; pretrained weights
are not bundled — see `DATA_PATHS.md`).

Images are produced by the companion
[`africa_morphology_imagery`](https://github.com/Sean-creater/africa_morphology_imagery)
repository. The full-Africa morphological indicator table used to apply
the trained model beyond this repository's evaluation set is produced by
the companion
[`africa_morphology_metrics`](https://github.com/Sean-creater/africa_morphology_metrics)
repository and is not bundled here; see `DATA_PATHS.md`.

### Full-Africa outputs

The morphological indicator table and embeddings for all African grid
cells are published as release assets:
[**Continent-scale morphology and embedding data**](https://github.com/Sean-creater/africa_morphology_embedding/releases/tag/africa-full-outputs-v1).
Both files join on `GridID`, consistent with the companion repositories.
See the manuscript's Methods and Supplementary Information for construction
and validation details. The release is private, visible only to repository
collaborators.

## Design validation

Six approaches to producing the embedding were compared, spanning a
spectrum from unsupervised linear projection to the teacher-student
approach used in this repository: no learning at all, direct supervised
learning on morphology indicators, a frozen or fine-tuned vision model on
imagery, and two knowledge-distillation variants (including the one used
throughout this repository). All six were evaluated against the same
held-out expert pairwise-similarity judgments.

The distillation approach used in this repository won on every evaluation
metric, and was the only one of the six that requires no imagery at
inference — the property that makes continent-scale deployment practical.
See the manuscript's Methods and Supplementary Information for the full
comparison, including each alternative's design and the evaluation
metrics used.

## Usage

```bash
# Train and extract the teacher (image-based) embedding
python methods/m4_finetuned_vit/train_triplet.py
python methods/m4_finetuned_vit/extract_triplet.py

# Train and extract the student (indicator-based) embedding
python methods/m6_triplet_distillation/train.py
python methods/m6_triplet_distillation/extract.py

# Reproduce the design-validation ranking table from the bundled outputs/
python eval_all_methods.py
```

See `methods/` for the remaining alternatives evaluated in design
validation.

## Repository layout

```
africa_morphology_embedding/
├── README.md, DATA_PATHS.md, environment.yml, requirements.txt, pyproject.toml
├── eval_all_methods.py
├── methods/                         # the teacher-student approach and five alternatives
├── shared/                          # shared path/dataset utilities
├── backbone/dinov3/                 # vendored, pruned vision backbone (weights not bundled)
├── data/
│   ├── annotations/                  # expert similarity-judgment dataset
│   ├── reference/                    # evaluation set + design-validation ranking table
│   ├── features/                     # morphology feature CSVs
│   └── example/                      # illustrative images
└── outputs/                          # final checkpoints + embeddings
```

## Citation

If you use this embedding model, please also cite the companion
morphology-extraction pipeline,
[`africa_morphology_metrics`](https://github.com/Sean-creater/africa_morphology_metrics),
which produces the indicator inputs used throughout.

## License

No license file was found in the source project this repository was derived
from. Until a license is chosen, all rights are reserved by default; add a
`LICENSE` file before distributing this code publicly. The vendored
`backbone/dinov3/` dependency is the one exception: it carries Meta's own
DINOv3 license (see `backbone/dinov3/LICENSE.md`), and its pretrained
weights (not bundled) are subject to the same license's terms.
