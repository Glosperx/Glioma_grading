# Label-free sub-region estimation and volumetric features for automated glioma grading

Code and notebooks behind the paper. Every table and figure in the manuscript is reproduced by one
of the seven notebooks, and each notebook states which section, table and figure it covers.

## What the pipeline does

1. A fine-tuned MedSAM2 segments the whole tumor slice by slice from the MRI and the per-slice
   masks are stacked into a volume; that volume is the only definition of the tumor the later
   stages see.
2. The predicted tumor is split into enhancing, necrotic and edema sub-regions by two thresholds
   on the contrast-enhancement map `T1ce - T1`, measured against normal-appearing brain, with no
   sub-region label of any kind.
3. Seventeen volumetric features are computed from the tumor and its sub-regions: four volumes,
   four sub-region fractions, and mean intensities on T1ce, on the enhancement map and on FLAIR.
   None of them use texture.
4. A classifier grades the case as high- or low-grade. Six standard architectures land between
   0.894 and **0.930** AUC on BraTS2020 (369 cases, 293 high-grade / 76 low-grade).
5. As an extension, a sixteen-component MedGemma description embedding is appended to the
   seventeen features and grading rises to **0.940**; the confidence interval on that gain reaches
   zero, and the repository reports it that way.

## Data

BraTS2020 (the Multimodal Brain Tumor Segmentation Challenge). **The dataset is not redistributed
here.** Obtain it from the challenge organisers, who require registration:
<https://www.med.upenn.edu/cbica/brats2020/data.html>.

The notebooks expect a working copy laid out as follows, pointed at by `GLIOMA_DATA_ROOT`:

```
$GLIOMA_DATA_ROOT/
  brats_npz_all/        one .npz per case with keys flair, t1, t1ce, t2, seg
  brats_seg_ft/         predicted whole-tumor masks, {case}_seg.npy, uint8, (155, 240, 240)
  brats_meta/           name_mapping.csv, which carries the grade
```

Volumes in `brats_npz_all/` are already clipped to their 1st and 99th percentiles and scaled to
8-bit; notebook 1 demonstrates this rather than assuming it, because applying the same transform a
second time would change the input the published fine-tune saw.

### What ships in `data_cache/`

Derived vectors, about 17 MB, no images and no masks: the seventeen-feature matrices, the two
BiomedCLIP embedding sets, the four MedGemma hidden-state sets with their answer files, MedGemma's
own grade-token confidences, and a two-column `cohort_labels.csv` of grades. With these, notebooks
2 to 6 run end to end with no BraTS download and no model weights, which covers every table in the
paper except the segmentation one.

`cohort_labels.csv` is a convenience derivative of the dataset's own `name_mapping.csv`; when a
real working copy is present that file is read instead.

## Environment

```bash
conda env create -f environment.yml && conda activate glioma
# or
python -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt
```

Two notes:

- **On macOS, `xgboost` must be imported before `torch`.** Both ship an OpenMP runtime, and with
  torch first xgboost segfaults at `fit` time rather than at import, which looks like a data
  problem instead of a linking one. `src/_openmp.py` fixes the order, and every module that
  touches torch imports it first, so notebook order does not matter.
- **MedSAM2 is a git dependency**, needed only by notebook 1:
  `pip install "git+https://github.com/bowang-lab/MedSAM2.git"`. The checkpoint
  (`wanglab/MedSAM2`, `MedSAM2_latest.pt`) and the configuration `sam2.1_hiera_t512.yaml` are
  downloaded on first use. `shap` and `matplotlib` are needed only for the figures.

Both paths are optional: `pip install -r requirements.txt` covers everything the table notebooks
need.

## Which notebook reproduces what

| Notebook | Paper sections | Tables | Figures | Runs from `data_cache/` |
|---|---|---|---|---|
| `01_segmentation_medsam2.ipynb` | 3.1, 4.1 | `tab:seg`, sub-region recovery | — | no, needs the volumes |
| `02_subregions_and_features.ipynb` | 3.2, 3.3, 4.2 | the seventeen features | — | yes |
| `03_grading_classifiers.ipynb` | 3.5, 3.6, 4.3 | `tab:clf`, the extended panel, the label-shuffle control | `fig:shap` | yes |
| `04_predicted_vs_groundtruth.ipynb` | 4.4 | `tab:abl` | — | yes |
| `05_foundation_embeddings.ipynb` | 3.4, 4.6 | `tab:prompts`, the two BiomedCLIP inputs | — | yes |
| `06_fusion_and_routing.ipynb` | 4.5 | `tab:errcorr`, `tab:matrix`, the routing result | — | yes |
| `07_figures.ipynb` | 4.1, 4.2 | — | `fig:segmentation`, `fig:subregions` | no, needs the volumes |

Shared code lives in `src/`: one implementation of the sub-region estimation, the features, the
cross-validation protocol, the fold-local PCA, the paired bootstrap and the table builders. The
notebooks assemble results; they do not restate the method.

## Provenance of the predicted masks

Three FLAIR fine-tunes exist in the history of this work, and only one produced the masks the
downstream tables read. This matters for anyone rerunning notebook 1:

| run | epochs | best held-out slice Dice | role |
|---|---|---|---|
| FLAIR | 12 | 0.8633 | produced `brats_seg_ft/`, which every later table reads |
| FLAIR | 6 | 0.8642 | the run behind the 0.76 to 0.86 figure |
| FLAIR, T1ce, T2 | 6 | 0.8644 | the second input method, reported as equivalent |

Notebook 1 reproduces the six-epoch recipe for both inputs and writes to its own directories; it
does not rebuild the twelve-epoch checkpoint bit for bit. Everything the paper reports from those
masks is therefore reproduced from the recipe rather than from a re-derived checkpoint, and the
shipped feature caches are the ones computed from the original masks.

Two points the paper states as limitations, repeated here because they bound what the numbers mean:

- The box prompts come from the reference masks, so the segmentation stage still needs an
  annotation.
- The cohort Dice of 0.913 is measured on cases the model was fine-tuned on. It describes what the
  grading stage receives, not performance on unseen data; the held-out figure is 0.86.

## Cohorts

Three cohorts appear in the paper, all three are correct, and they must not be reconciled:

- **369 cases** for grading and every fusion table, with no exclusions;
- **368 cases** for segmentation quality, because one case carries no reference mask, so there is
  nothing to score against;
- **368 cases** for the predicted-versus-ground-truth comparison, for the same reason.

A fourth, 367, appears only as a wiring proof: an earlier run of this pipeline lost one case to an
unreadable file and skipped it silently. Notebooks 3 and 4 reproduce that cohort first to show the
protocol has not drifted, then report the corrected one.

## Hardware and run times

Measured on an Apple laptop using the Metal backend:

| stage | cost |
|---|---|
| MedSAM2 fine-tune | ~0.169 s per iteration, about **55 minutes for six epochs** |
| MedSAM2 inference over 369 cases | a few hours, one forward pass per tumor-bearing slice |
| MedGemma, one prompt over 369 cases | ~7 s per case, ~12 s for the reasoning prompt |
| BiomedCLIP, 369 regions | 3 to 4 minutes |
| every grading, fusion and prompt table | CPU only, minutes, from `data_cache/` |

Two Metal-specific details are handled in the code: the fused attention kernel selector is
CUDA-only, so plain scaled dot-product attention is enabled instead; and MedGemma needs its cache
released after each case or a long loop dies partway through.

## Reproduction status

Verified by executing the notebooks against the manuscript: `tab:clf`, `tab:abl`, `tab:errcorr`,
`tab:matrix`, `tab:prompts`, the label-shuffle control and the routing tables all reproduce their
published values. Each of those notebooks carries a cell that prints the computed value beside the
published one and flags any row that drifts.

`tab:seg`, the sub-region recovery percentages and the two qualitative figures were **not** re-run
in preparing this release, because they need the full BraTS working copy and, for the segmentation
table, a fresh fine-tune. Their code is the same code that produced the published numbers.

## License

Code: MIT, see `LICENSE`. BraTS2020 and the pretrained MedSAM2, BiomedCLIP and MedGemma weights
remain under their own terms.
