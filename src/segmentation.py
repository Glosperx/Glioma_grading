"""Whole-tumor segmentation quality and ground-truth sub-region recovery.

Scores the predicted masks the rest of the pipeline consumes against the reference masks, per
case, over the 368 cases that carry a reference (paper Section 4.1, tab:seg).

Two different things are measured and the paper keeps them apart:

    overlap   Dice, IoU, precision and recall of the whole tumor
    recovery  how much of each reference sub-region falls inside the predicted mask, which
              is what says the segmentation loses edema rather than enhancing tumor
"""
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from . import config
from .subregions import EPS

METRIC_NAMES = ["Dice", "IoU", "Precision", "Recall"]
SUB_REGIONS = ["ET", "NCR", "ED"]
SUB_REGION_LABELS = {"ET": "Enhancing tumor (ET)", "NCR": "Necrotic core (NCR)",
                     "ED": "Edema (ED)"}


def overlap_metrics(predicted, reference):
    """Dice, IoU, precision and recall of one predicted mask against one reference."""
    intersection = float((predicted & reference).sum())
    p, g = float(predicted.sum()), float(reference.sum())
    return (
        2 * intersection / (p + g + EPS),
        intersection / (float((predicted | reference).sum()) + EPS),
        intersection / (p + EPS),
        intersection / (g + EPS),
    )


def scan_masks(mask_dirs, progress=True):
    """One pass over the volumes, scoring every named mask directory at once.

    Cases with no reference mask are skipped, which is what makes this cohort 368 rather
    than 369. Recovery is accumulated two ways: averaged over cases, and pooled over voxels,
    because a per-case mean weights a small tumor as heavily as a large one.
    """
    metrics = {key: [] for key in mask_dirs}
    recovery = {key: {r: [] for r in SUB_REGIONS} for key in mask_dirs}
    pooled = {key: {r: [0.0, 0.0] for r in SUB_REGIONS} for key in mask_dirs}
    case_ids = []

    for path in tqdm(sorted(config.NPZ_DIR.glob("*.npz")), desc="Scoring masks",
                     disable=not progress):
        case_id = path.stem
        data = np.load(str(path), allow_pickle=True)
        if "seg" not in data.files:
            continue
        seg = data["seg"]
        reference = seg > 0
        sub = {"ET": seg == 4, "NCR": seg == 1, "ED": seg == 2}

        row = {}
        for key, mask_dir in mask_dirs.items():
            predicted = np.load(str(mask_dir / f"{case_id}_seg.npy")).astype(bool)
            row[key] = overlap_metrics(predicted, reference)
            for name, mask in sub.items():
                total = float(mask.sum())
                if total == 0:
                    continue
                hit = float((mask & predicted).sum())
                recovery[key][name].append(hit / total)
                pooled[key][name][0] += hit
                pooled[key][name][1] += total
        for key in mask_dirs:
            metrics[key].append(row[key])
        case_ids.append(case_id)

    return case_ids, {k: np.array(v) for k, v in metrics.items()}, recovery, pooled


def segmentation_table(metrics, key, grade=None):
    """tab:seg -- mean, standard deviation and the value the worst 5% fall below."""
    rows = []
    for i, name in enumerate(METRIC_NAMES):
        column = metrics[key][:, i]
        rows.append({
            "Metric": name,
            "Mean": round(float(column.mean()), 3),
            "Std": round(float(column.std()), 3),
            "Worst 5%": round(float(np.percentile(column, 5)), 3),
        })
    table = pd.DataFrame(rows).set_index("Metric")
    if grade is not None:
        dice = metrics[key][:, 0]
        for label, code in [("Dice, high-grade", "HGG"), ("Dice, low-grade", "LGG")]:
            subset = dice[grade == code]
            table.loc[label] = [round(float(subset.mean()), 3),
                                round(float(subset.std()), 3),
                                round(float(np.percentile(subset, 5)), 3)]
    return table


def recovery_table(recovery, pooled, keys):
    """How much of each reference sub-region sits inside the predicted mask, as a percentage.

    The paper quotes the per-case column. Where two mask sets are given, the difference
    between them is added so the two fine-tuning inputs can be compared directly.
    """
    rows = []
    for name in SUB_REGIONS:
        row = {"Sub-region": SUB_REGION_LABELS[name]}
        for key in keys:
            per_case = np.array(recovery[key][name])
            hit, total = pooled[key][name]
            row[f"{key}, per case %"] = round(100 * float(per_case.mean()), 1)
            row[f"{key}, pooled %"] = round(100 * hit / total, 1)
            row[f"{key}, n"] = len(per_case)
        if len(keys) == 2:
            row["difference, per case"] = round(
                100 * (np.array(recovery[keys[1]][name]).mean()
                       - np.array(recovery[keys[0]][name]).mean()), 2)
        rows.append(row)
    return pd.DataFrame(rows).set_index("Sub-region")


def worst_cases(case_ids, metrics, key, grade=None, n=6):
    """The lowest-Dice cases, which is how the qualitative figure picks its third row."""
    dice = metrics[key][:, 0]
    order = np.argsort(dice)[:n]
    rows = [{"Case": case_ids[j],
             "Grade": (grade[j] if grade is not None else ""),
             "Dice": round(float(dice[j]), 3)} for j in order]
    return pd.DataFrame(rows).set_index("Case")


def case_recovery(case_id, mask_dir=None):
    """Per-sub-region recovery for a single case, for the sentence about the weakest case."""
    mask_dir = mask_dir or config.SEG_DIR
    data = np.load(str(config.NPZ_DIR / f"{case_id}.npz"), allow_pickle=True)
    seg = data["seg"]
    predicted = np.load(str(mask_dir / f"{case_id}_seg.npy")).astype(bool)
    sub = {"ET": seg == 4, "NCR": seg == 1, "ED": seg == 2}
    out = {"Dice": round(overlap_metrics(predicted, seg > 0)[0], 3)}
    core = sub["ET"] | sub["NCR"]
    out["ET + NCR kept %"] = round(100 * float((core & predicted).sum())
                                   / (float(core.sum()) + EPS), 1)
    for name, mask in sub.items():
        total = float(mask.sum())
        out[f"{name} kept %"] = (round(100 * float((mask & predicted).sum()) / total, 1)
                                 if total else float("nan"))
    return out
