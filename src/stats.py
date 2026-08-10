"""Comparison statistics: the paired bootstrap and the error-complementarity measures.

Everything here operates on pooled out-of-fold scores, so two feature sets are always
compared on the same folds and the same cases (paper Section 3.6).
"""
from decimal import ROUND_HALF_UP, Decimal

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import matthews_corrcoef, roc_auc_score, roc_curve

N_BOOTSTRAP = 2000
SEED = 42


def round_half_up(value, decimals=3):
    """Round with a half step going away from zero, the convention the paper prints.

    Python rounds a binary-inexact half towards even, so a bootstrap bound of -0.0145 becomes
    -0.014 while the manuscript shows -0.015. Displaying the published convention keeps the
    notebook output and the tables identical instead of differing in the last digit.
    """
    quantum = Decimal(1).scaleb(-decimals)
    return float(Decimal(repr(float(value))).quantize(quantum, rounding=ROUND_HALF_UP))


def paired_bootstrap_delta(y, scores_base, scores_augmented, n_boot=N_BOOTSTRAP, seed=SEED):
    """Resample cases, not folds, and recompute both AUCs on each resample.

    Returns the mean difference, a 95% confidence interval and the fraction of resamples in
    which the augmented set wins. A resample that happens to hold one grade only carries no
    AUC and is skipped.
    """
    rng = np.random.default_rng(seed)
    n = len(y)
    deltas = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y[idx])) < 2:
            continue
        deltas.append(roc_auc_score(y[idx], scores_augmented[idx])
                      - roc_auc_score(y[idx], scores_base[idx]))
    deltas = np.asarray(deltas)
    return {
        "mean": float(deltas.mean()),
        "ci_low": float(np.percentile(deltas, 2.5)),
        "ci_high": float(np.percentile(deltas, 97.5)),
        "p_positive": float((deltas > 0).mean()),
        "n_resamples": int(deltas.size),
    }


#: Precision the bootstrap is reported at before any further rounding.
NATIVE_DECIMALS = 4


def format_ci(stats, decimals=NATIVE_DECIMALS):
    """The interval as the paper prints it.

    The bootstrap is a four-decimal quantity, and the manuscript's three-decimal intervals were
    rounded from that four-decimal form rather than from the raw bound: +0.009453 is reported as
    +0.0095 and then as +0.010. Rounding in the same two steps is what makes a three-decimal
    interval here identical to the published one.
    """
    def render(value):
        value = round_half_up(value, NATIVE_DECIMALS)
        return round_half_up(value, decimals) if decimals < NATIVE_DECIMALS else value

    return f"[{render(stats['ci_low']):+.{decimals}f}, {render(stats['ci_high']):+.{decimals}f}]"


# ---------------------------------------------------------------------------
# Operating point and per-case errors
# ---------------------------------------------------------------------------
def youden_threshold(y, scores):
    """Operating point maximising true-positive minus false-positive rate."""
    fpr, tpr, thresholds = roc_curve(y, scores)
    return float(thresholds[np.argmax(tpr - fpr)])


def wrong_mask(y, scores, threshold):
    """Boolean per case: did this feature set get the grade wrong at that threshold."""
    return (scores >= threshold).astype(int) != y


def residuals(y, scores):
    """Signed per-case error, positive when the score is too low for the true grade."""
    return y - scores


# ---------------------------------------------------------------------------
# Error complementarity (paper Section 4.5, tab:errcorr)
# ---------------------------------------------------------------------------
def within_class_pearson(a, b, y):
    """Mean of the residual correlations computed inside each grade.

    Measuring within a grade removes the confound that both methods look "more wrong" on the
    minority class, which would otherwise inflate the correlation. This is the number the
    paper reports; a low value means the two methods fail on different cases.
    """
    return float(np.mean([pearsonr(a[y == c], b[y == c]).statistic for c in (0, 1)]))


def within_class_detail(a, b, y):
    """The same quantity split by grade, with Spearman alongside Pearson."""
    out = {}
    for cls, name in [(0, "high-grade"), (1, "low-grade")]:
        m = y == cls
        out[name] = {
            "n": int(m.sum()),
            "pearson": float(pearsonr(a[m], b[m]).statistic),
            "spearman": float(spearmanr(a[m], b[m]).statistic),
        }
    return out


def decision_phi(wrong_a, wrong_b):
    """Matthews correlation of two right/wrong masks: do the same cases come out wrong."""
    return float(matthews_corrcoef(wrong_a.astype(int), wrong_b.astype(int)))


# ---------------------------------------------------------------------------
# Upper bounds on what a per-case fusion could reach
# ---------------------------------------------------------------------------
def oracle_rank(y, *score_vectors):
    """Per-case best of N, pushing every true low-grade up and every high-grade down.

    A tight upper bound on the AUC of any per-case router or weighted average over these
    score vectors. It reads the labels, so it is not achievable; it exists to show whether
    the headroom a router would chase is real.
    """
    stacked = np.vstack(score_vectors)
    return np.where(y == 1, stacked.max(0), stacked.min(0))


def pair_rescue(y, scores_base, scores_other):
    """AUC is a pair statistic, so count pairs rather than cases.

    Of the (low-grade, high-grade) pairs the base method misranks, how many does the other
    method rank correctly, and how many correct pairs does it break.
    """
    pos, neg = scores_base[y == 1][:, None], scores_base[y == 0][None, :]
    pos_o, neg_o = scores_other[y == 1][:, None], scores_other[y == 0][None, :]
    base_wrong = pos <= neg                       # ties count as misranked
    other_right = pos_o > neg_o
    n_base_wrong = int(base_wrong.sum())
    return {
        "n_pairs": int(base_wrong.size),
        "base_misranked": n_base_wrong,
        "rescued": int((base_wrong & other_right).sum()),
        "rescue_rate": (int((base_wrong & other_right).sum()) / n_base_wrong
                        if n_base_wrong else float("nan")),
        "broken": int((~base_wrong & ~other_right).sum()),
    }
