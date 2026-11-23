"""The evaluation protocol shared by every experiment (paper Section 3.6).

Case-level five-fold stratified cross-validation, one row per case, scored by AUC. Every
scaler and every PCA lives inside the pipeline, so it is fit on training folds only.

Two levels of AUC appear in the paper and they are not interchangeable:

    mean of per-fold AUCs   ``evaluate`` / ``cross_validate`` -- tab:clf, tab:abl
    pooled out-of-fold AUC  ``out_of_fold`` + ``roc_auc_score`` -- tab:errcorr, tab:matrix,
                            tab:prompts

The pooled form is the conservative one and is what the paired bootstrap resamples, so the
fusion and prompt tables use it. Which one a table uses is stated in its caption.
"""
from . import _openmp  # noqa: F401  keep xgboost ahead of torch on macOS

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import (GradientBoostingClassifier, HistGradientBoostingClassifier,
                              RandomForestClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (StratifiedKFold, cross_val_predict, cross_validate)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .stats import round_half_up

SEED = 42

#: The one cross-validation object every table is scored with. Its folds are also the folds of
#: the segmentation cross-fit (``data_cache/folds.csv``).
CV = StratifiedKFold(5, shuffle=True, random_state=SEED)

#: Fold assignments of the repeated cross-validation. Seed 42 is the single run every other
#: table reports; the other four only reshuffle which cases share a fold.
REPEAT_SEEDS = (42, 1, 2, 3, 4)

#: Components each foundation-model embedding is reduced to, fit inside the folds.
EMBED_PCA_DIM = 16

#: Blocks that pass through scaled instead of being reduced. The seventeen volumetric
#: features are already low-dimensional, so PCA would only discard information.
NO_PCA_BLOCKS = {"volumetric"}

#: The panel of tab:clf, with the labels the paper prints.
CLASSIFIERS = {
    "SVM (RBF)": lambda: SVC(kernel="rbf", probability=True,
                             class_weight="balanced", random_state=SEED),
    "SVM (linear)": lambda: SVC(kernel="linear", probability=True,
                                class_weight="balanced", random_state=SEED),
    "Random forest": lambda: RandomForestClassifier(n_estimators=300, max_depth=6,
                                                    class_weight="balanced",
                                                    random_state=SEED),
    "Gradient boosting": lambda: GradientBoostingClassifier(random_state=SEED),
    "Logistic reg.": lambda: LogisticRegression(max_iter=1000, class_weight="balanced"),
    "$k$-NN ($k = 5$)": lambda: KNeighborsClassifier(n_neighbors=5),
}

#: The model the fusion tables lead with, fixed before any delta was looked at.
PRIMARY_MODEL = "Random forest"

SCORING = ["accuracy", "precision", "recall", "f1", "roc_auc"]


def seeded_cv(seed):
    """The protocol's cross-validation with another fold assignment, for the repeated runs."""
    return StratifiedKFold(5, shuffle=True, random_state=seed)


def make_xgboost(y_train):
    """XGBoost with the class imbalance handled by ``scale_pos_weight``."""
    from xgboost import XGBClassifier

    n_neg = int((y_train == 0).sum())
    n_pos = int((y_train == 1).sum())
    return XGBClassifier(
        n_estimators=300, max_depth=3, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
        scale_pos_weight=n_neg / max(n_pos, 1),
        eval_metric="logloss", random_state=SEED,
    )


def wide_classifiers(y):
    """The tab:clf panel extended, as described in Section 3.5.

    The original six stay unchanged so the extension is additive.
    """
    panel = dict(CLASSIFIERS)
    panel["XGBoost"] = lambda: make_xgboost(y)
    panel["Hist. grad. boosting"] = lambda: HistGradientBoostingClassifier(
        max_depth=3, learning_rate=0.05, max_iter=300, random_state=SEED)
    panel["Logistic reg. ($L_2$)"] = lambda: LogisticRegression(
        C=0.1, max_iter=2000, class_weight="balanced")
    panel["Logistic reg. (elastic net)"] = lambda: LogisticRegression(
        penalty="elasticnet", solver="saga", l1_ratio=0.5, C=0.1,
        max_iter=4000, class_weight="balanced")
    return panel


# ---------------------------------------------------------------------------
# Design matrices and fold-local pipelines
# ---------------------------------------------------------------------------
def build_design(blocks, names):
    """Concatenate the named blocks and record which columns each one owns."""
    parts, spans, start = [], [], 0
    for name in names:
        block = blocks[name]
        parts.append(block)
        spans.append((name, start, start + block.shape[1]))
        start += block.shape[1]
    return np.hstack(parts), spans


def make_pipeline_for(spans, build_clf):
    """One pipeline per feature set: scale the volumetric block, reduce every embedding.

    The ColumnTransformer sits inside the Pipeline, which is what keeps each PCA fit
    fold-local. Fitting PCA on the full matrix first is the leak that flatters an
    embedding, so it is worth being explicit about.
    """
    transformers = []
    for name, start, stop in spans:
        if name in NO_PCA_BLOCKS:
            transformers.append((name, StandardScaler(), slice(start, stop)))
        else:
            transformers.append((
                f"pca_{name}",
                make_pipeline(StandardScaler(),
                              PCA(n_components=EMBED_PCA_DIM, random_state=SEED)),
                slice(start, stop),
            ))
    return Pipeline([("blocks", ColumnTransformer(transformers)),
                     ("clf", build_clf())])


def scaled_pipeline(build_clf):
    """Scaler plus classifier, for a matrix that needs no block handling."""
    return make_pipeline(StandardScaler(), build_clf())


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
def evaluate(X, y, include_xgboost=True, classifiers=None, decimals=3):
    """The classifier panel under the fixed protocol, as a DataFrame.

    Columns are means over the five folds. Accuracy, F1 and AUC are what tab:clf prints;
    precision and recall are kept because they show the panel is not trading one for the
    other on the minority class.

    ``decimals=None`` keeps full precision, which matters when a difference between two
    columns is reported: rounding each column first and subtracting afterwards moves the last
    digit, so tab:abl subtracts before it rounds.
    """
    panel = dict(classifiers or CLASSIFIERS)
    if include_xgboost and "XGBoost" not in panel:
        panel["XGBoost"] = lambda: make_xgboost(y)

    def maybe_round(value):
        return value if decimals is None else round_half_up(value, decimals)

    rows = []
    for name, build in panel.items():
        scores = cross_validate(scaled_pipeline(build), X, y, cv=CV, scoring=SCORING)
        rows.append({
            "Classifier": name,
            "Accuracy": maybe_round(scores["test_accuracy"].mean()),
            "Precision": maybe_round(scores["test_precision"].mean()),
            "Recall": maybe_round(scores["test_recall"].mean()),
            "F1": maybe_round(scores["test_f1"].mean()),
            "AUC": maybe_round(scores["test_roc_auc"].mean()),
        })
    return pd.DataFrame(rows).set_index("Classifier")


def block_auc(blocks, names, y, build_clf=None, cv=None):
    """Mean-of-folds AUC for a feature set assembled from blocks."""
    X, spans = build_design(blocks, names)
    build = build_clf or CLASSIFIERS[PRIMARY_MODEL]
    scores = cross_validate(make_pipeline_for(spans, build), X, y,
                            cv=cv or CV, scoring="roc_auc")
    return float(scores["test_score"].mean())


def out_of_fold(blocks, names, y, build_clf=None, cv=None):
    """Pooled out-of-fold P(low-grade) for a feature set assembled from blocks.

    Every arm of a comparison is scored on identical folds, which is what makes the paired
    bootstrap of ``stats.paired_bootstrap_delta`` valid.
    """
    X, spans = build_design(blocks, names)
    build = build_clf or CLASSIFIERS[PRIMARY_MODEL]
    return out_of_fold_scores(X, y, make_pipeline_for(spans, build), cv=cv)


def out_of_fold_scores(X, y, pipeline, cv=None):
    """Pooled out-of-fold positive-class probabilities for an assembled pipeline."""
    return cross_val_predict(pipeline, X, y, cv=cv or CV, method="predict_proba")[:, 1]


def shuffled_label_control(X, y, seed=SEED, include_xgboost=False):
    """Integrity check: with permuted grades every AUC must collapse towards chance."""
    rng = np.random.default_rng(seed)
    return evaluate(X, rng.permutation(y), include_xgboost=include_xgboost)
