"""The per-case routing experiment (paper Section 4.5, the negative result).

Concatenation lets the classifier weigh the two views globally. The alternative is per case:
send a case to MedGemma's own grade when the model is confident, and to the volumetric
classifier otherwise. This module builds those routers and the diagnostic that explains why
none of them works.

The order matters. A router can only help if some label-free signal separates the cases where
deferring *rescues* the volumetric classifier from the cases where it *breaks* it. That
separation is measured directly by ``flip_separation``; if it sits at chance, no gate built
from that signal can work, whatever the deferral fraction.
"""
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

from . import stats

#: Fractions of the cohort handed to the other view, from none to half.
DEFERRAL_FRACTIONS = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50]


def confidence(scores, threshold=0.5):
    """Distance from the decision threshold: how sure this view is about a case."""
    return np.abs(scores - threshold)


def percentile(values):
    """Values on a common [0, 1] scale, so two confidences can be compared at all.

    A volumetric probability and a language-model token probability live on different scales,
    so comparing them raw would be meaningless.
    """
    return rankdata(values) / len(values)


def load_native_confidence(path, cases):
    """MedGemma's own P(low-grade), ordered to the cohort.

    Recovered by teacher-forcing the saved answer up to the assessment field and reading the
    next-token distribution, so it is the probability the model actually decided with.
    """
    cached = np.load(path, allow_pickle=True)
    by_case = {str(c): float(v) for c, v in zip(cached["case_ids"], cached["p_lgg_native"])}
    missing = [c for c in cases if c not in by_case]
    if missing:
        raise ValueError(f"{len(missing)} cases lack a native confidence (e.g. {missing[:3]})")
    return np.array([by_case[c] for c in cases])


def saturation(p_native, tolerance=0.01):
    """How peaked the native confidence is. A saturated signal cannot gate anything."""
    margin = np.minimum(p_native, 1 - p_native)
    return pd.DataFrame(
        {"value": [round(float(np.median(confidence(p_native))), 3),
                   int((margin < tolerance).sum()), len(p_native)]},
        index=[f"median distance from indecision (0.5 is maximal uncertainty)",
               f"cases within {tolerance} of a hard 0 or 1",
               "cases"])


def standalone_table(y, named_scores):
    """Is each grade signal any good on its own, before any routing is attempted."""
    rows = []
    for label, (scores, threshold) in named_scores.items():
        predicted = (scores >= threshold).astype(int)
        rows.append({"Grade signal": label,
                     "AUC": stats.round_half_up(roc_auc_score(y, scores), 3),
                     f"Correct of {len(y)}": int((predicted == y).sum())})
    return pd.DataFrame(rows).set_index("Grade signal")


def deferral_sweep(y, p_base, threshold_base, p_other, threshold_other, gate,
                   fractions=None, most_confident_first=True):
    """Hand a growing share of cases to the other view, ordered by the gate.

    ``fixes`` counts deferred cases the base view had wrong and the other gets right;
    ``breaks`` counts the reverse. The blended score keeps each view's ranking inside its own
    region, so the column is an AUC and not just an accuracy.
    """
    fractions = fractions or DEFERRAL_FRACTIONS
    n = len(y)
    wrong_base = stats.wrong_mask(y, p_base, threshold_base)
    wrong_other = stats.wrong_mask(y, p_other, threshold_other)
    predicted_base = (p_base >= threshold_base).astype(int)
    predicted_other = (p_other >= threshold_other).astype(int)
    rank_base, rank_other = rankdata(p_base), rankdata(p_other)

    order = np.argsort(-gate if most_confident_first else gate)
    rows, blended_by_fraction = [], {}
    for fraction in fractions:
        k = int(round(fraction * n))
        deferred = np.zeros(n, bool)
        deferred[order[:k]] = True
        routed = np.where(deferred, predicted_other, predicted_base)
        blended = np.where(deferred, rank_other, rank_base)
        blended_by_fraction[fraction] = blended
        fixes = int((deferred & wrong_base & ~wrong_other).sum())
        breaks = int((deferred & ~wrong_base & wrong_other).sum())
        rows.append({
            "Deferred": f"{int(round(fraction * 100))}%",
            "Cases": k,
            "Fixes": fixes,
            "Breaks": breaks,
            "Net correct": fixes - breaks,
            "Balanced acc.": stats.round_half_up(balanced_accuracy_score(y, routed), 3),
            "AUC": stats.round_half_up(roc_auc_score(y, blended), 3),
        })
    return pd.DataFrame(rows).set_index("Deferred"), blended_by_fraction


def best_router_bootstrap(y, p_base, sweep_table, blended_by_fraction, label):
    """Bootstrap the sweep's best point against the base view.

    Picking the best fraction after seeing the results is post-hoc and flatters the router, so
    it is the most generous test available; the point is that even this does not win.
    """
    positive = sweep_table[sweep_table.index != "0%"]
    best = positive["Net correct"].idxmax()
    fraction = float(best.rstrip("%")) / 100
    blended = blended_by_fraction[min(blended_by_fraction,
                                      key=lambda f: abs(f - fraction))]
    bs = stats.paired_bootstrap_delta(y, p_base, blended)
    return pd.DataFrame([{
        "Router": label,
        "Best fraction (post-hoc)": best,
        "$\\Delta$ AUC": f"{roc_auc_score(y, blended) - roc_auc_score(y, p_base):+.4f}",
        "95\\% CI": stats.format_ci(bs),
        "P($\\Delta > 0$)": stats.round_half_up(bs["p_positive"], 3),
    }]).set_index("Router")


def flip_separation(y, p_base, threshold_base, p_other, threshold_other, signals):
    """The decisive diagnostic: can any label-free signal tell a rescue from a break?

    Restricted to the cases where the two views disagree, since those are the only ones a
    router can change. The target is whether the base view is wrong there. A signal that could
    drive a router scores far from 0.5; at 0.5 it carries no routing information at all.

    One signal scores well and is useless: the volumetric probability itself. Rescuable and
    breakable are *defined* by whether the volumetric view is wrong, which that probability
    sets, so it is circular, and routing on it collapses onto the majority grade.
    """
    predicted_base = (p_base >= threshold_base).astype(int)
    predicted_other = (p_other >= threshold_other).astype(int)
    flip = predicted_base != predicted_other
    rescuable = stats.wrong_mask(y, p_base, threshold_base)[flip].astype(int)

    rows = [{"Label-free signal over the disagreement cases": label,
             "AUC (rescuable vs. breakable)": stats.round_half_up(
                 roc_auc_score(rescuable, signal[flip]), 3)}
            for label, signal in signals.items()]
    summary = {"disagreements": int(flip.sum()),
               "rescuable": int(rescuable.sum()),
               "breakable": int(flip.sum() - rescuable.sum())}
    return pd.DataFrame(rows).set_index("Label-free signal over the disagreement cases"), summary
