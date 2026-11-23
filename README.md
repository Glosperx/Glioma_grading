# Label-free sub-region estimation and volumetric features for automated glioma grading

Code and notebooks behind the paper *From Pixels to Grade: Label-Free Sub-Region Estimation and
Volumetric Features for Automated Glioma Grading* (ICAART 2027). Every table and figure in the
manuscript is reproduced by one of the seven notebooks, and each notebook states which section,
table and figure it covers.

## What the pipeline does

1. A fine-tuned MedSAM2 segments the whole tumor slice by slice from the MRI and the per-slice
   masks are stacked into a volume; that volume is the only definition of the tumor the later
   stages see. The masks are cross-fitted on the grading folds, so no case is graded from a mask
   predicted by a model that saw it during fine-tuning.
2. The predicted tumor is split into enhancing, necrotic and edema sub-regions by two thresholds
   on the contrast-enhancement map `T1ce - T1`, measured against normal-appearing brain, with no
   sub-region label of any kind.
3. Seventeen volumetric features are computed from the tumor and its sub-regions: four volumes,
   four sub-region fractions, and mean intensities on T1ce, on the enhancement map and on FLAIR.
   None of them use texture.
4. A classifier grades the case as high- or low-grade. Six standard architectures land between
   0.882 and **0.927** AUC on BraTS2020 (369 cases, 293 high-grade / 76 low-grade).
5. As an extension, a sixteen-component MedGemma description embedding, from the few-shot
   prompt, is appended to the seventeen features, and the random forest rises from 0.925 to
   **0.937**, with a 95% interval of [+0.001, +0.025] on the gain. That is one assignment of cases
   to folds. Over five assignments the mean gain is +0.0035, positive on four of the five, and the
   plain MedGemma prompt gains at least as much (+0.0052). The gain is small, and the repository
   reports it that way.

## Data

BraTS2020 (the Multimodal Brain Tumor Segmentation Challenge). **The dataset is not redistributed
here.** Obtain it from the challenge organisers, who require registration:
<https://www.med.upenn.edu/cbica/brats2020/data.html>.

The notebooks expect a working copy laid out as follows, pointed at by `GLIOMA_DATA_ROOT`:

```
$GLIOMA_DATA_ROOT/
  brats_npz_all/        one .npz per case with keys flair, t1, t1ce, t2, seg
  brats_seg_cf/         cross-fitted whole-tumor masks, {case}_seg.npy, uint8, (155, 240, 240)
  brats_seg_mc/         optional: masks of the FLAIR, T1ce, T2 fine-tune, for notebook 1
  brats_meta/           name_mapping.csv, which carries the grade
```

Volumes in `brats_npz_all/` are already clipped to their 1st and 99th percentiles and scaled to
8-bit; notebook 1 demonstrates this rather than assuming it, because applying the same transform a
second time would change the input the fine-tune saw.

### What ships in `data_cache/`

Derived vectors, about 17 MB, no images, no masks and no model weights. Everything that depends on
the predicted masks comes from the cross-fitted ones:

- `features_pred17.npz`, the seventeen features from the cross-fitted masks;
- `biomedclip_roi_embeddings.npz` and `biomedclip_enhancement_embeddings.npz`, the two BiomedCLIP
  inputs;
- the four MedGemma hidden-state sets with their answer files (`medgemma_hidden_states.npz` and
  `medgemma_t1ce_answers.csv` for the baseline prompt, then `medgemma_numeric_*`,
  `medgemma_fewshot_*` and `medgemma_fewshot_cot_*`), and MedGemma's own grade-token confidences
  in `medgemma_native_conf.npz`;
- `folds.csv`, the fold of every case, shared by the segmentation cross-fit and every grading
  table;
- `crossfit_training.csv`, the record of the five fine-tunes (training and test cases, the
  test-fold slice Dice, the training time);
- `features_gt17.npz`, the same seventeen features from the ground-truth sub-regions, and
  `features_mc17.npz`, from the masks of the FLAIR, T1ce, T2 fine-tune; neither depends on the
  cross-fit;
- `cohort_labels.csv`, a two-column table of grades derived from the dataset's own
  `name_mapping.csv`, which is read instead whenever a working copy is present.

Every embedding is computed from one image per case, built on the slice with the largest predicted
tumor area. For BiomedCLIP and for the baseline MedGemma prompt, the cases whose encoder input is
unchanged by the cross-fit keep their earlier vector, computed from the identical image, and every
other case was re-extracted (337 of 369 for BiomedCLIP, 368 of 369 for MedGemma). The few-shot,
numeric and reasoning prompts and the grade-token confidences were run on all 369 cases. Re-running
MedGemma on samples of cases reproduced the stored vectors exactly.

With these, notebooks 2 to 6 run end to end with no BraTS download and no model weights, which
covers every table in the paper except the segmentation one.

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
| `01_segmentation_medsam2.ipynb` | 3.1, 4.1 | `tab:seg`, sub-region recovery, per-fold held-out Dice | — | the per-fold Dice only; the rest needs the volumes |
| `02_subregions_and_features.ipynb` | 3.2, 3.3, 4.2 | the seventeen features | — | yes |
| `03_grading_classifiers.ipynb` | 3.5, 3.6, 4.3 | `tab:clf`, the extended panel, the label-shuffle control | — | yes |
| `04_predicted_vs_groundtruth.ipynb` | 4.4 | `tab:abl` | — | yes |
| `05_foundation_embeddings.ipynb` | 3.4, 4.6 | `tab:prompts`, the numeric single-field AUCs, the two BiomedCLIP inputs | — | yes |
| `06_fusion_and_routing.ipynb` | 4.5 | `tab:errcorr`, `tab:matrix`, the five-seed mean, the routing result | — | yes |
| `07_figures.ipynb` | 4.1, 4.2, 4.3 | — | `fig:segmentation`, `fig:subregions`, `fig:shap` | `fig:shap` only |

Shared code lives in `src/`: one implementation of the sub-region estimation, the features, the
cross-validation protocol, the fold-local PCA, the paired bootstrap and the table builders. The
notebooks assemble results; they do not restate the method.

## How the masks were produced

The grading protocol is five-fold stratified cross-validation, `StratifiedKFold(5, shuffle=True,
random_state=42)` over the sorted case list. The segmentation is cross-fitted on those same five
folds: for each fold, MedSAM2 is fine-tuned on the cases of the other four folds with the FLAIR
input for twelve epochs, the final-epoch weights are kept with no checkpoint selection, and only
the cases of that fold are segmented with it. `cohort.check_folds` asserts case by case that
`folds.csv` is the fold assignment the grading cross-validation produces.

Scored once after training, the five held-out folds reach a slice-level Dice of 0.843 to 0.867,
0.858 on average, which is the 0.86 the paper quotes. Over each full volume, the cross-fitted masks
reach a Dice of 0.898 across the 368 cases that carry a reference mask.

Notebook 1 also keeps the comparison of the two inputs: FLAIR alone and FLAIR, T1ce and T2, each
fine-tuned for six epochs on a single 80/20 case split, reach the same best held-out slice Dice
(0.8642 and 0.8644). The second input was not cross-fitted, so its masks are in-sample for the
cases of its training split, and the paper only reports it as equivalent.

Two points the paper states as limitations, repeated here because they bound what the numbers
mean:

- The box prompts come from the reference masks, so the segmentation stage still needs an
  annotation.
- The MedGemma few-shot prompt was chosen after the earlier results were known (see below).

## The MedGemma prompts

Four prompt formulations are compared in `tab:prompts`, and the one the paper calls "MedGemma
few-shot" is the few-shot prompt in `src/encoders.py`. The numeric, few-shot and reasoning prompts
open with the same description of the two image panels. As first submitted, that description said
normal brain appears mid-grey on the enhancement map; on the panel the model is actually shown,
normal brain and the empty background are dark grey. The three prompts correct that one clause
and change nothing else (`encoders.SUBMISSION_CLAUSE` and `encoders.CORRECTED_CLAUSE`); the
baseline prompt never contained it. The few-shot prompt, this correction included, was written
after earlier results were known, as the paper's limitations state.

## Cohorts

Three cohorts appear in the paper, all three are correct, and they must not be reconciled:

- **369 cases** for grading and every fusion table, with no exclusions;
- **368 cases** for segmentation quality, because one case carries no reference mask, so there is
  nothing to score against;
- **368 cases** for the predicted-versus-ground-truth comparison, for the same reason.

A fourth, 367, appears only as a wiring proof: an earlier run of this pipeline lost one case to an
unreadable file and skipped it silently. Notebook 4 reproduces that cohort's ground-truth column
first, which no mask can change, to show the protocol has not drifted, then reports the corrected
cohort.

## Hardware and run times

Measured on an Apple laptop using the Metal backend:

| stage | cost |
|---|---|
| MedSAM2 cross-fit, five folds of twelve epochs | about 100 minutes of training per fold, 8 h 49 min in all with inference |
| MedSAM2 input comparison | ~0.169 s per iteration, about 55 minutes for six epochs |
| MedGemma, one prompt over 369 cases | ~7 to 8 s per case, ~12 s for the reasoning prompt |
| MedGemma grade-token confidence | ~1.5 s per case |
| BiomedCLIP, 369 regions | 3 to 4 minutes |
| every grading, fusion and prompt table | CPU only, minutes, from `data_cache/` |

Two Metal-specific details are handled in the code: the fused attention kernel selector is
CUDA-only, so plain scaled dot-product attention is enabled instead; and MedGemma needs its cache
released after each case or a long loop dies partway through.

## Reproduction status

Executed for this release:

- **Notebooks 2 to 6**, end to end from `data_cache/` alone, with no BraTS copy and no model
  weights. Every value they print that the paper also prints is checked by a cell that compares
  the two as the paper prints them, and all match: `tab:clf`, the extended panel, the label-shuffle
  control, `tab:abl`, `tab:errcorr`, `tab:matrix`, the five-seed means, `tab:prompts`, the
  numeric single-field AUCs, the answer counts, the two BiomedCLIP inputs and the routing tables.
  With the BraTS copy present, notebook 2 also rebuilds both feature matrices from the masks and
  reproduces the shipped ones exactly.
- **Notebook 1**, the evaluation cells, against the BraTS copy and the cross-fitted masks:
  `tab:seg`, the sub-region recovery, the weakest case and the comparison of the two inputs. The
  training and inference cells were not re-run: they need the volumes and about nine hours on an
  Apple GPU, and the masks they would write are the ones the original cross-fit run produced with
  the same recipe.
- **Notebook 7**: `fig:segmentation` and `fig:subregions` are reproduced byte for byte. `fig:shap`
  reproduces the same attributions exactly; its beeswarm draws the dots in a random order, so the
  image matches in content rather than pixel for pixel.

Not re-run for this release: the generation cells of notebook 5, which call MedGemma and BiomedCLIP
to produce the caches in `data_cache/`.

The version of this repository that accompanied the submission, with masks from a single
fine-tune that had seen most of the cases it segmented (cohort Dice 0.913, grading 0.930 and 0.940),
remains in its history.

## License

Code: MIT, see `LICENSE`. BraTS2020 and the pretrained MedSAM2, BiomedCLIP and MedGemma weights
remain under their own terms.
