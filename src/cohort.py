"""Cohort assembly: grades, feature matrices and the foundation-model blocks.

Three cohorts appear in the paper and all three are correct. They must not be harmonised:

    n = 369   grading (tab:clf) and every fusion table -- no exclusions
    n = 368   segmentation (tab:seg) -- one case carries no reference mask, so there is
              nothing to score against
    n = 368   predicted versus ground-truth sub-regions (tab:abl) -- same cause, the
              comparison needs a ground-truth mask on both sides

A fourth cohort, n = 367, appears only as a wiring proof on the ground-truth arm of tab:abl.
An earlier run of this pipeline lost one case to an unreadable copy of its ``.npz`` and silently
skipped it; reproducing that cohort shows the protocol is unchanged, and the corrected cohort is
what the paper reports.

The five grading folds are also the folds of the segmentation cross-fit: the masks of every
fold were predicted by a MedSAM2 fine-tuned on the other four. ``check_folds`` asserts that the
shipped fold file and the grading cross-validation agree case by case.
"""
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from . import config
from .subregions import FEATURE_NAMES, estimate_subregions, features, groundtruth_subregions

#: The case whose reference mask is absent, so it cannot enter tab:seg and tab:abl.
NO_GROUND_TRUTH = "BraTS20_Training_355"

#: The case an earlier run lost to an unreadable file. It reads without error and belongs
#: in the cohort; it is named only so the wiring proof can put it back out.
PREVIOUSLY_SKIPPED = "BraTS20_Training_004"

HIGH_GRADE, LOW_GRADE = "HGG", "LGG"


# ---------------------------------------------------------------------------
# Grades
# ---------------------------------------------------------------------------
def load_grade_labels():
    """Map case identifier to grade. High-grade encodes to 0, low-grade to 1.

    Reads the dataset's own ``name_mapping.csv`` when the BraTS working copy is present, and
    otherwise the two-column table shipped in the cache directory, so the grading and fusion
    notebooks run without downloading the images.
    """
    path = config.META_DIR / "name_mapping.csv"
    if path.exists():
        meta = pd.read_csv(path)
        meta.columns = [c.strip() for c in meta.columns]
        id_col = next(c for c in meta.columns if "BraTS_2020" in c or "brats20" in c.lower())
        grade_col = next(c for c in meta.columns if "grade" in c.lower())
    elif config.COHORT_LABELS.exists():
        meta = pd.read_csv(config.COHORT_LABELS)
        id_col, grade_col = "case_id", "grade"
    else:
        raise FileNotFoundError(
            f"No grades found. Expected {path} or {config.COHORT_LABELS}.")
    return dict(zip(meta[id_col].astype(str).str.strip(),
                    meta[grade_col].astype(str).str.strip()))


def encode_grades(grade_map, cases):
    """Label vector in the order of ``cases``: 0 high-grade, 1 low-grade."""
    return np.array([0 if grade_map[c] == HIGH_GRADE else 1 for c in cases])


# ---------------------------------------------------------------------------
# Folds shared by the segmentation cross-fit and the grading cross-validation
# ---------------------------------------------------------------------------
def load_folds():
    """``case_id, grade, fold`` for the 369 cases, in sorted case order."""
    return pd.read_csv(config.FOLDS)


def check_folds(cases, y):
    """Assert that the fold of every case is the test fold the grading CV puts it in.

    The cross-fitted masks are only out of sample for the grading folds if the two fold
    assignments are the same, so this is checked rather than assumed.
    """
    from .evaluation import CV

    folds = load_folds()
    assert folds["case_id"].tolist() == list(cases), "fold file and cohort differ in order"
    assert (folds["grade"].map({HIGH_GRADE: 0, LOW_GRADE: 1}).to_numpy() == y).all()
    fold = np.full(len(cases), -1)
    for k, (_, test) in enumerate(CV.split(np.zeros((len(y), 1)), y)):
        fold[test] = k
    assert (fold == folds["fold"].to_numpy()).all(), "fold file disagrees with the grading CV"
    return folds


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------
def _write_cache(path, by_case):
    case_ids = sorted(by_case)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path,
             features=np.stack([by_case[c] for c in case_ids]),
             case_ids=np.array(case_ids),
             colnames=np.array(FEATURE_NAMES))
    return by_case


def _read_cache(path):
    cached = np.load(path, allow_pickle=True)
    return {str(cid): vec for cid, vec in zip(cached["case_ids"], cached["features"])}


def extract_features(seg_dir=None, cache=None, force=False, progress=True):
    """Seventeen features per case from predicted masks, with the sub-regions estimated.

    Reads the cache when it is there, which is what lets the grading notebooks run with no
    BraTS copy at all. Extraction from the volumes takes a few minutes.
    """
    seg_dir = seg_dir or config.SEG_DIR
    cache = cache or config.FEATURES_PRED
    if cache.exists() and not force:
        return _read_cache(cache)

    npz_by_case = {p.stem: p for p in sorted(config.NPZ_DIR.glob("*.npz"))}
    seg_by_case = {p.stem.replace("_seg", ""): p for p in sorted(seg_dir.glob("*.npy"))}
    print(f"{len(npz_by_case)} volumes, {len(seg_by_case)} predicted masks")

    by_case, skipped = {}, {"unreadable": [], "no mask": [], "shape mismatch": []}
    for case_id in tqdm(sorted(npz_by_case), desc="Features", disable=not progress):
        try:
            data = np.load(str(npz_by_case[case_id]), allow_pickle=True)
            flair = data["flair"].astype(np.float32)
            t1 = data["t1"].astype(np.float32)
            t1ce = data["t1ce"].astype(np.float32)
        except Exception:
            skipped["unreadable"].append(case_id)
            continue
        if case_id not in seg_by_case:
            skipped["no mask"].append(case_id)
            continue
        WT = np.load(str(seg_by_case[case_id])).astype(bool)
        if WT.shape != flair.shape:
            skipped["shape mismatch"].append(case_id)
            continue

        ET, NCR, ED = estimate_subregions(WT, t1ce, t1, flair)
        by_case[case_id] = features(ET, NCR, ED, WT, t1ce, t1, flair)

    print(f"extracted {len(by_case)} cases; " +
          ", ".join(f"{k}: {len(v)}" for k, v in skipped.items()))
    return _write_cache(cache, by_case)


def extract_gt_features(cache=None, force=False, progress=True):
    """Seventeen features per case from the ground-truth sub-regions.

    Nothing is estimated: the sub-regions come off the dataset's label codes and go into the
    same ``features`` the predicted arm uses, so the two arms differ only in their masks.
    """
    cache = cache or config.FEATURES_GT
    if cache.exists() and not force:
        return _read_cache(cache)

    by_case, no_seg = {}, []
    for path in tqdm(sorted(config.NPZ_DIR.glob("*.npz")), desc="Ground-truth features",
                     disable=not progress):
        data = np.load(str(path), allow_pickle=True)
        if "seg" not in data.files:
            no_seg.append(path.stem)
            continue
        flair = data["flair"].astype(np.float32)
        t1 = data["t1"].astype(np.float32)
        t1ce = data["t1ce"].astype(np.float32)
        ET, NCR, ED, WT = groundtruth_subregions(data["seg"])
        by_case[path.stem] = features(ET, NCR, ED, WT, t1ce, t1, flair)

    print(f"extracted {len(by_case)} cases; no reference mask: {no_seg}")
    return _write_cache(cache, by_case)


def build_matrix(feature_by_case, grade_map, exclude=frozenset()):
    """Design matrix, labels and case list over the cases present in both inputs."""
    cases = sorted(c for c in feature_by_case if c in grade_map and c not in exclude)
    X = np.stack([feature_by_case[c] for c in cases])
    return X, encode_grades(grade_map, cases), cases


# ---------------------------------------------------------------------------
# Foundation-model blocks
# ---------------------------------------------------------------------------
def _by_case(path, array_key):
    cached = np.load(path, allow_pickle=True)
    return {str(cid): vec for cid, vec in zip(cached["case_ids"], cached[array_key])}


def load_blocks(verbose=True):
    """The volumetric block plus the two encoder blocks, on the cases present in all three.

    The intersection is what fixes n = 369 for every fusion table. Later blocks are added
    with ``inject`` and must cover exactly these cases, so the folds stay identical across
    the whole of Section 4.5.
    """
    feature_by_case = extract_features()
    grade_map = load_grade_labels()
    clip_by_case = _by_case(config.BIOMEDCLIP_T1CE, "embeddings")
    gemma_by_case = _by_case(config.MEDGEMMA_HIDDEN["baseline"], "hidden")

    cases = sorted(c for c in feature_by_case
                   if c in grade_map and c in clip_by_case and c in gemma_by_case)
    blocks = {
        "volumetric": np.stack([feature_by_case[c] for c in cases]),
        "medgemma": np.stack([gemma_by_case[c] for c in cases]),
        "biomedclip_t1ce": np.stack([clip_by_case[c] for c in cases]),
    }
    y = encode_grades(grade_map, cases)
    if verbose:
        print(f"cohort n={len(cases)}  high-grade={int((y == 0).sum())}  "
              f"low-grade={int(y.sum())}")
        print("  " + "  ".join(f"{k}={v.shape}" for k, v in blocks.items()))
    return blocks, y, cases


def inject(blocks, cases, key, path, array_key="embeddings"):
    """Add a cached block, ordered to the cohort. Refuses partial coverage.

    A block that silently dropped cases would be scored on a different cohort, which is the
    one way these tables can quietly stop being comparable.
    """
    by_case = _by_case(path, array_key)
    missing = [c for c in cases if c not in by_case]
    if missing:
        raise ValueError(f"{path.name}: {len(missing)} of {len(cases)} cases missing "
                         f"(e.g. {missing[:3]})")
    blocks[key] = np.stack([by_case[c] for c in cases])
    return blocks


def inject_encoder_blocks(blocks, cases, keys=("medgemma_fewshot", "biomedclip_enh")):
    """Add the blocks Section 4.5 needs beyond the two loaded by ``load_blocks``."""
    sources = {
        "medgemma_fewshot": (config.MEDGEMMA_HIDDEN["fewshot"], "hidden"),
        "medgemma_numeric": (config.MEDGEMMA_HIDDEN["numeric"], "hidden"),
        "medgemma_fewshot_cot": (config.MEDGEMMA_HIDDEN["fewshot_cot"], "hidden"),
        "biomedclip_enh": (config.BIOMEDCLIP_ENH, "embeddings"),
    }
    for key in keys:
        path, array_key = sources[key]
        inject(blocks, cases, key, path, array_key)
    return blocks
