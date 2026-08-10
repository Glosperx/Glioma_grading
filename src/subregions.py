"""Label-free sub-region estimation and the seventeen volumetric features.

Stage two and stage three of the pipeline (paper Sections 3.2 and 3.3). Both work from
the predicted whole-tumor mask and never read a sub-region label.

The thresholds and the feature order are load-bearing: they define the numbers in the
paper's tables. Changing anything in this module breaks comparability.
"""
import numpy as np

EPS = 1e-6

#: Feature order for every design matrix in the study. Four volumes, four sub-region
#: fractions of the tumor volume, then mean intensities on T1ce, on the enhancement map
#: and on FLAIR, the last two normalized by healthy tissue.
FEATURE_NAMES = [
    "V_WT", "V_ET", "V_NCR", "V_ED",
    "ET_frac", "NCR_frac", "ED_frac", "ET_core_frac",
    "T1ce_WT", "T1ce_ET", "T1ce_NCR",
    "enh_WT", "enh_ET",
    "FLAIR_WT", "FLAIR_ED",
    "enh_ratio_ET", "edema_ratio",
]

#: Grouping used only for display, so a notebook can show what the seventeen are.
FEATURE_GROUPS = {
    "volume": ["V_WT", "V_ET", "V_NCR", "V_ED"],
    "sub-region fraction": ["ET_frac", "NCR_frac", "ED_frac", "ET_core_frac"],
    "mean intensity": ["T1ce_WT", "T1ce_ET", "T1ce_NCR", "enh_WT", "enh_ET",
                       "FLAIR_WT", "FLAIR_ED"],
    "normalized by healthy tissue": ["enh_ratio_ET", "edema_ratio"],
}


def mean_in(img, mask):
    """Mean of ``img`` over ``mask``, zero when the mask is empty."""
    return float(img[mask].mean()) if mask.any() else 0.0


def healthy_reference(WT, flair):
    """Normal-appearing brain outside the predicted tumor.

    ``flair > 10`` drops the air around the skull-stripped brain. If that leaves almost
    nothing, fall back to everything outside the tumor.
    """
    healthy = (flair > 10) & (~WT)
    if healthy.sum() < 50:
        healthy = ~WT
    return healthy


def estimate_subregions(WT, t1ce, t1, flair):
    """Split the predicted tumor into enhancing, necrotic and edema, with no labels.

    Enhancement map ``enh = T1ce - T1`` is bright wherever tissue takes up contrast. Its
    mean and standard deviation over healthy brain give the reference for what normal
    tissue looks like, and two thresholds then label every voxel inside the mask:

        ET  = enh  > mu_enh  + 1.5 * sigma_enh
        NCR = T1ce < mu_t1ce - 0.5 * sigma_t1ce      (and not already ET)
        ED  = the rest of the mask

    ET takes precedence over NCR. The three sets partition ``WT`` exactly.
    """
    empty = np.zeros_like(WT, bool)
    if WT.sum() == 0:
        return empty, empty, empty

    enh = t1ce.astype(np.float32) - t1.astype(np.float32)
    healthy = healthy_reference(WT, flair)
    e_mean, e_std = float(enh[healthy].mean()), float(enh[healthy].std()) + EPS
    t_mean, t_std = float(t1ce[healthy].mean()), float(t1ce[healthy].std()) + EPS

    ET = WT & (enh > e_mean + 1.5 * e_std)
    NCR = WT & (t1ce < t_mean - 0.5 * t_std) & (~ET)
    ED = WT & (~ET) & (~NCR)
    return ET, NCR, ED


def groundtruth_subregions(seg):
    """Sub-regions read straight off the BraTS label codes. Nothing is estimated.

    Returns ``(ET, NCR, ED, WT)`` for the ground-truth arm of the Section 4.4 ablation.
    """
    return (seg == 4), (seg == 1), (seg == 2), (seg > 0)


def features(ET, NCR, ED, WT, t1ce, t1, flair):
    """The seventeen features, in ``FEATURE_NAMES`` order.

    Voxel counts stand in for volumes: BraTS is resampled to an isotropic 1 mm grid, so a
    count is a volume in cubic millimetres up to a constant the classifiers absorb.
    """
    healthy = ~WT
    mean_healthy_t1ce = mean_in(t1ce, healthy) + EPS
    mean_healthy_flair = mean_in(flair, healthy) + EPS
    enh = t1ce.astype(np.float32) - t1.astype(np.float32)

    V_WT, V_ET = float(WT.sum()), float(ET.sum())
    V_NCR, V_ED = float(NCR.sum()), float(ED.sum())

    return np.array([
        V_WT, V_ET, V_NCR, V_ED,
        V_ET / (V_WT + EPS), V_NCR / (V_WT + EPS), V_ED / (V_WT + EPS),
        V_ET / (V_ET + V_NCR + EPS),
        mean_in(t1ce, WT), mean_in(t1ce, ET), mean_in(t1ce, NCR),
        mean_in(enh, WT), mean_in(enh, ET),
        mean_in(flair, WT), mean_in(flair, ED),
        mean_in(t1ce, ET) / mean_healthy_t1ce,
        mean_in(flair, ED) / mean_healthy_flair,
    ], dtype=np.float32)


def enhancement_sigma(t1ce, t1, WT, flair):
    """Enhancement map expressed in standard deviations of healthy brain.

    The display form of the same quantity the thresholds use, for the figures and for the
    panels handed to MedGemma.
    """
    enh = t1ce.astype(np.float32) - t1.astype(np.float32)
    healthy = healthy_reference(WT, flair)
    return (enh - enh[healthy].mean()) / (enh[healthy].std() + EPS)


def self_test(seed=0):
    """Assert the invariants the pipeline relies on, without any real data.

    The three sub-regions must partition the mask exactly and be mutually disjoint, and a
    feature vector must hold seventeen finite numbers.
    """
    rng = np.random.default_rng(seed)
    depth, height, width = 12, 48, 48

    flair = rng.normal(60, 10, (depth, height, width)).astype(np.float32)
    t1 = rng.normal(60, 10, (depth, height, width)).astype(np.float32)
    t1ce = t1.copy()
    t1ce[4:6, 18:26, 18:26] += 60.0          # a bright enhancing nodule

    WT = np.zeros((depth, height, width), bool)
    WT[4:8, 16:32, 16:32] = True

    ET, NCR, ED = estimate_subregions(WT, t1ce, t1, flair)
    assert np.array_equal(ET | NCR | ED, WT), "sub-regions must cover the mask exactly"
    assert not (ET & NCR).any() and not (ET & ED).any() and not (NCR & ED).any(), \
        "sub-regions must be disjoint"

    vector = features(ET, NCR, ED, WT, t1ce, t1, flair)
    assert vector.shape == (17,), f"expected 17 features, got {vector.shape}"
    assert np.isfinite(vector).all(), "feature vector holds a NaN or an infinity"
    assert len(FEATURE_NAMES) == 17

    empty = np.zeros_like(WT, bool)
    assert all(m.sum() == 0 for m in estimate_subregions(empty, t1ce, t1, flair)), \
        "an empty mask must yield empty sub-regions"

    return {"ET": int(ET.sum()), "NCR": int(NCR.sum()), "ED": int(ED.sum()),
            "WT": int(WT.sum()), "n_features": int(vector.shape[0])}
