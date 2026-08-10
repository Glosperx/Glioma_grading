"""Paths and run-wide constants.

Two roots, both overridable by environment variable, both relative by default so the
release contains no machine-specific path:

    GLIOMA_DATA_ROOT   the BraTS2020 working copy (volumes, predicted masks, metadata).
                       Not redistributed; see the README for the expected layout.
    GLIOMA_CACHE_DIR   derived artifacts that ship with this repository (feature
                       matrices, encoder embeddings, model answers).

Notebooks import from here and never build a path of their own.
"""
import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Roots
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]

DATA_ROOT = Path(os.environ.get("GLIOMA_DATA_ROOT", REPO_ROOT / "data"))
CACHE_DIR = Path(os.environ.get("GLIOMA_CACHE_DIR", REPO_ROOT / "data_cache"))

# ---------------------------------------------------------------------------
# BraTS working copy
# ---------------------------------------------------------------------------
NPZ_DIR = DATA_ROOT / "brats_npz_all"      # one .npz per case: flair, t1, t1ce, t2, seg
SEG_DIR = DATA_ROOT / "brats_seg_ft"       # predicted whole-tumor masks, {case}_seg.npy
SEG_MC_DIR = DATA_ROOT / "brats_seg_mc"    # masks from the three-channel fine-tune
META_DIR = DATA_ROOT / "brats_meta"        # name_mapping.csv carries the HGG/LGG grade

# ---------------------------------------------------------------------------
# Derived caches shipped with the repository
# ---------------------------------------------------------------------------
FEATURES_PRED = CACHE_DIR / "features_pred17.npz"   # 17 features, predicted sub-regions
FEATURES_GT = CACHE_DIR / "features_gt17.npz"       # 17 features, ground-truth sub-regions
FEATURES_MC = CACHE_DIR / "features_mc17.npz"       # 17 features, three-channel masks

BIOMEDCLIP_T1CE = CACHE_DIR / "biomedclip_roi_embeddings.npz"
BIOMEDCLIP_ENH = CACHE_DIR / "biomedclip_enhancement_embeddings.npz"

MEDGEMMA_HIDDEN = {
    "baseline": CACHE_DIR / "medgemma_hidden_states.npz",
    "numeric": CACHE_DIR / "medgemma_numeric_hidden.npz",
    "fewshot": CACHE_DIR / "medgemma_fewshot_hidden.npz",
    "fewshot_cot": CACHE_DIR / "medgemma_fewshot_cot_hidden.npz",
}
MEDGEMMA_ANSWERS = {
    "baseline": CACHE_DIR / "medgemma_t1ce_answers.csv",
    "numeric": CACHE_DIR / "medgemma_numeric_answers.csv",
    "fewshot": CACHE_DIR / "medgemma_fewshot_answers.csv",
    "fewshot_cot": CACHE_DIR / "medgemma_fewshot_cot_answers.csv",
}
MEDGEMMA_NATIVE_CONF = CACHE_DIR / "medgemma_native_conf.npz"

COHORT_LABELS = CACHE_DIR / "cohort_labels.csv"     # fallback grades, case_id,grade

FIGURE_DIR = Path(os.environ.get("GLIOMA_FIGURE_DIR", REPO_ROOT / "figures"))

# ---------------------------------------------------------------------------
# Pretrained model identifiers
# ---------------------------------------------------------------------------
MEDSAM2_REPO, MEDSAM2_CHECKPOINT = "wanglab/MedSAM2", "MedSAM2_latest.pt"
MEDSAM2_CONFIG = "configs/sam2.1_hiera_t512.yaml"
BIOMEDCLIP_ID = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
MEDGEMMA_ID = "google/medgemma-1.5-4b-it"


def describe() -> str:
    """One block a notebook can print to show what it resolved and what exists."""
    rows = [
        ("DATA_ROOT", DATA_ROOT, DATA_ROOT.exists()),
        ("  brats_npz_all", NPZ_DIR, NPZ_DIR.exists()),
        ("  brats_seg_ft", SEG_DIR, SEG_DIR.exists()),
        ("  brats_meta", META_DIR, META_DIR.exists()),
        ("CACHE_DIR", CACHE_DIR, CACHE_DIR.exists()),
    ]
    width = max(len(name) for name, _, _ in rows)
    lines = [
        f"{name:<{width}}  {'present' if ok else 'MISSING':>7}  {path}"
        for name, path, ok in rows
    ]
    return "\n".join(lines)
