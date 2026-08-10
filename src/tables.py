"""The paper's tables, assembled from the other modules.

Each function returns a DataFrame whose rows and columns carry the labels the paper prints, so
a notebook cell and a table in the manuscript can be compared line by line.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from . import cohort, stats
from .evaluation import (CLASSIFIERS, PRIMARY_MODEL, evaluate, out_of_fold, wide_classifiers)

DASH = "---"

#: Rounding used for every published AUC.
DECIMALS = 3


def _rounded_delta(auc, auc_baseline, decimals=DECIMALS):
    """Difference between two AUCs as the paper states it.

    tab:errcorr and tab:matrix both define their difference column against the *printed*
    baseline ("the standalone AUC minus the volumetric baseline of 0.930"), so the subtraction
    happens after rounding and the column always agrees with the two AUCs beside it. tab:abl
    is the exception: it subtracts at full precision and rounds afterwards. Following each
    table's own definition is what makes the reproduction exact.
    """
    return (stats.round_half_up(auc, decimals)
            - stats.round_half_up(auc_baseline, decimals))


#: Row order and block keys for the two Section 4.5 tables.
COMPONENTS = [
    ("Volumetric features", ["volumetric"]),
    ("MedGemma", ["medgemma"]),
    ("MedGemma (few-shot)", ["medgemma_fewshot"]),
    ("BiomedCLIP (T1ce)", ["biomedclip_t1ce"]),
    ("BiomedCLIP (enhancement)", ["biomedclip_enh"]),
]

FEATURE_SETS = [
    ("Volumetric features", ["volumetric"]),
    ("$+$ MedGemma", ["volumetric", "medgemma"]),
    ("$+$ MedGemma (few-shot)", ["volumetric", "medgemma_fewshot"]),
    ("$+$ BiomedCLIP (T1ce)", ["volumetric", "biomedclip_t1ce"]),
    ("$+$ BiomedCLIP (enhancement)", ["volumetric", "biomedclip_enh"]),
    ("$+$ MedGemma $+$ BiomedCLIP (T1ce)",
     ["volumetric", "medgemma", "biomedclip_t1ce"]),
    ("$+$ MedGemma (few-shot) $+$ BiomedCLIP (enhancement)",
     ["volumetric", "medgemma_fewshot", "biomedclip_enh"]),
]

PROMPTS = [
    ("Baseline (categorical)", "medgemma"),
    ("Numeric (percentages)", "medgemma_numeric"),
    ("Few-shot", "medgemma_fewshot"),
    ("Few-shot $+$ reasoning", "medgemma_fewshot_cot"),
]


# ---------------------------------------------------------------------------
# tab:clf -- grading classifiers
# ---------------------------------------------------------------------------
def table_clf(X, y, include_xgboost=True):
    """The six-classifier panel as the paper prints it, plus the XGBoost row."""
    full = evaluate(X, y, include_xgboost=include_xgboost)
    return full[["Accuracy", "F1", "AUC"]]


def table_clf_extended(X, y):
    """The panel extended with XGBoost, histogram gradient boosting and regularized
    logistic regression (Section 4.3)."""
    full = evaluate(X, y, include_xgboost=False, classifiers=wide_classifiers(y))
    return full[["Accuracy", "F1", "AUC"]]


# ---------------------------------------------------------------------------
# tab:abl -- predicted versus ground-truth sub-regions
# ---------------------------------------------------------------------------
def table_abl(feature_gt, feature_pred, grade_map, exclude=frozenset()):
    """Both arms on the same matched cases, so the differences are paired.

    Evaluating the predicted arm on its own full cohort would break the pairing and make the
    differences meaningless. Sign convention: predicted minus ground truth.
    """
    matched = sorted(c for c in feature_pred
                     if c in feature_gt and c in grade_map and c not in exclude)
    y = cohort.encode_grades(grade_map, matched)
    X_gt = np.stack([feature_gt[c] for c in matched])
    X_pred = np.stack([feature_pred[c] for c in matched])

    # Full precision on both arms, so the difference is taken before either is rounded.
    ground_truth = evaluate(X_gt, y, include_xgboost=False, decimals=None)["AUC"]
    predicted = evaluate(X_pred, y, include_xgboost=False, decimals=None)["AUC"]
    table = pd.DataFrame({
        "Ground truth": ground_truth.map(stats.round_half_up),
        "Predicted": predicted.map(stats.round_half_up),
        "$\\Delta$": (predicted - ground_truth).map(stats.round_half_up),
    })
    table.index.name = "Classifier"
    return table, matched, y


# ---------------------------------------------------------------------------
# tab:errcorr -- standalone AUC and error correlation with the volumetric features
# ---------------------------------------------------------------------------
def component_scores(blocks, y, components=None):
    """Pooled out-of-fold scores for each component method, computed once and reused."""
    return {label: out_of_fold(blocks, keys, y)
            for label, keys in (components or COMPONENTS)}


def table_errcorr(blocks, y, scores=None, baseline="Volumetric features"):
    """Each component alone, its gap to the volumetric baseline, and its error correlation.

    The correlation is measured inside each grade, so agreement driven by the grade itself
    does not inflate it. A low value means the two methods fail on different cases.
    """
    scores = scores or component_scores(blocks, y)
    auc = {label: roc_auc_score(y, p) for label, p in scores.items()}
    residual = {label: stats.residuals(y, p) for label, p in scores.items()}

    rows = []
    for label, _ in COMPONENTS:
        if label == baseline:
            rows.append({"Method": label,
                         "AUC alone": stats.round_half_up(auc[label], DECIMALS),
                         "$\\Delta$ vs. Vol.": DASH, "Err. corr. vs. Vol.": DASH})
            continue
        rows.append({
            "Method": label,
            "AUC alone": stats.round_half_up(auc[label], DECIMALS),
            "$\\Delta$ vs. Vol.": f"{_rounded_delta(auc[label], auc[baseline]):+.3f}",
            "Err. corr. vs. Vol.": stats.round_half_up(
                stats.within_class_pearson(residual[baseline], residual[label], y), 3),
        })
    return pd.DataFrame(rows).set_index("Method")


def errcorr_matrices(blocks, y, scores=None):
    """The full pairwise picture behind tab:errcorr, three ways.

    pooled            Pearson of the signed errors over all cases
    within grade      the same, averaged inside each grade -- the confound-free number
    decision level    Matthews correlation of the right/wrong masks, that is, whether the
                      same cases come out wrong
    """
    scores = scores or component_scores(blocks, y)
    labels = [label for label, _ in COMPONENTS]
    residual = {label: stats.residuals(y, scores[label]) for label in labels}
    threshold = {label: stats.youden_threshold(y, scores[label]) for label in labels}
    wrong = {label: stats.wrong_mask(y, scores[label], threshold[label]) for label in labels}

    def matrix(fn):
        M = np.array([[1.0 if a == b else fn(a, b) for b in labels] for a in labels])
        return pd.DataFrame(np.round(M, 3), index=labels, columns=labels)

    from scipy.stats import pearsonr
    return {
        "pooled": matrix(lambda a, b: pearsonr(residual[a], residual[b]).statistic),
        "within grade": matrix(
            lambda a, b: stats.within_class_pearson(residual[a], residual[b], y)),
        "decision level": matrix(lambda a, b: stats.decision_phi(wrong[a], wrong[b])),
    }


# ---------------------------------------------------------------------------
# tab:matrix -- fusion by concatenation, with a paired bootstrap
# ---------------------------------------------------------------------------
def table_matrix(blocks, y, feature_sets=None, baseline_keys=("volumetric",), detail=False):
    """Grading AUC as each embedding is appended to the seventeen volumetric features.

    Every row is the same random forest on a longer input vector, and every difference is
    resampled against the volumetric baseline over cases.

    ``detail=True`` reports the difference, the interval and the probability at four decimals
    instead of the paper's three, which is where the borderline nature of the one positive row
    is easiest to read.
    """
    feature_sets = feature_sets or FEATURE_SETS
    p_base = out_of_fold(blocks, list(baseline_keys), y)
    auc_base = roc_auc_score(y, p_base)
    decimals = 4 if detail else 3

    rows = []
    for label, keys in feature_sets:
        p = out_of_fold(blocks, keys, y)
        auc = roc_auc_score(y, p)
        if list(keys) == list(baseline_keys):
            rows.append({"Feature set": label, "AUC": stats.round_half_up(auc, DECIMALS),
                         "$\\Delta$ vs. Volumetric": DASH, "95\\% CI": DASH,
                         "P($\\Delta > 0$)": DASH})
            continue
        bs = stats.paired_bootstrap_delta(y, p_base, p)
        delta = auc - auc_base if detail else _rounded_delta(auc, auc_base)
        rows.append({
            "Feature set": label,
            "AUC": stats.round_half_up(auc, DECIMALS),
            "$\\Delta$ vs. Volumetric": f"{delta:+.{decimals}f}",
            "95\\% CI": stats.format_ci(bs, decimals=decimals),
            "P($\\Delta > 0$)": stats.round_half_up(bs["p_positive"], 3),
        })
    return pd.DataFrame(rows).set_index("Feature set")


# ---------------------------------------------------------------------------
# tab:prompts -- the four MedGemma prompt formulations
# ---------------------------------------------------------------------------
def table_prompts(blocks, y, volumetric_label="Volumetric features",
                  baseline_prompt="Baseline (categorical)"):
    """Each prompt's description embedding scored on its own, pooled out of fold.

    Same protocol as tab:errcorr, so the baseline and few-shot rows are the same numbers that
    table reports. A prompt is judged by what its hidden state encodes, not by the literal
    answer it writes.
    """
    auc_vol = roc_auc_score(y, out_of_fold(blocks, ["volumetric"], y))
    auc = {label: roc_auc_score(y, out_of_fold(blocks, [key], y)) for label, key in PROMPTS}

    rows = []
    for label, _ in PROMPTS:
        rows.append({
            "Prompt": label,
            "Standalone AUC": stats.round_half_up(auc[label], DECIMALS),
            "$\\Delta$ vs. baseline": (
                DASH if label == baseline_prompt
                else f"{_rounded_delta(auc[label], auc[baseline_prompt]):+.3f}"),
            "$\\Delta$ vs. Vol.": f"{_rounded_delta(auc[label], auc_vol):+.3f}",
        })
    return pd.DataFrame(rows).set_index("Prompt")


# ---------------------------------------------------------------------------
# Helpers a notebook uses to check itself against the manuscript
# ---------------------------------------------------------------------------
def compare_to_paper(table, column, expected, tolerance=0.001):
    """Put the published value beside the computed one and flag any row that drifts.

    Reporting the drift is the point; nothing here corrects either side.
    """
    check = pd.DataFrame({
        "computed": table[column],
        "paper": pd.Series(expected),
    })
    check["difference"] = (check["computed"] - check["paper"]).round(4)
    check["match"] = check["difference"].abs() <= tolerance
    return check
