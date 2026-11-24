# From Pixels to Grade: Label-Free Sub-Region Estimation and Volumetric Features for Automated Glioma Grading

Code for the ICAART 2027 paper.

## Overview

1. A fine-tuned MedSAM2 segments the whole tumor.
2. The tumor is split into enhancing, necrotic and edema sub-regions by thresholding the
   contrast-enhancement map (T1ce − T1), with no ground-truth sub-region labels.
3. Seventeen volumetric features are computed and passed to a classifier.
4. Optionally, a MedGemma description embedding is appended to the features.

## Results

BraTS2020, 369 cases, five-fold stratified cross-validation, with segmentation masks cross-fitted
on the same folds.

| | |
|---|---|
| Whole-tumor Dice | 0.898 |
| SVM (RBF), volumetric features | 0.927 AUC |
| Random forest, volumetric features | 0.925 AUC |
| Random forest, volumetric features + MedGemma (few-shot prompt) | 0.937 AUC |

## Installation

```bash
conda env create -f environment.yml && conda activate glioma
# or
pip install -r requirements.txt
```

Notebook 1 also needs MedSAM2:

```bash
pip install "git+https://github.com/bowang-lab/MedSAM2.git"
```

## Data

The experiments use the BraTS2020 dataset, which is not included in this repository. To convert
the training data into the format the notebooks read:

```bash
python prepare_brats.py /path/to/MICCAI_BraTS2020_TrainingData
```

This writes to `GLIOMA_DATA_ROOT` (default `data/`):

```
$GLIOMA_DATA_ROOT/
  brats_npz_all/   one .npz per case with keys flair, t1, t1ce, t2, seg
  brats_meta/      name_mapping.csv
  brats_seg_cf/    cross-fitted whole-tumor masks, written by notebook 1
```

`data_cache/` holds the precomputed features and embeddings, so notebooks 2 to 6 run without the
dataset or any model weights.

## Notebooks

| Notebook | Reproduces | Needs BraTS |
|---|---|---|
| `01_segmentation_medsam2` | MedSAM2 fine-tuning, Table 1 | yes |
| `02_subregions_and_features` | sub-regions and features | no |
| `03_grading_classifiers` | Table 2 | no |
| `04_predicted_vs_groundtruth` | Table 3 | no |
| `05_foundation_embeddings` | Table 6 | no |
| `06_fusion_and_routing` | Tables 4 and 5 | no |
| `07_figures` | Figures 2, 3 and 4 | yes, except Figure 4 |

## License

MIT. BraTS2020 and the pretrained MedSAM2, BiomedCLIP and MedGemma weights remain under their own
terms.
