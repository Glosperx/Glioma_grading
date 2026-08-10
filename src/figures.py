"""Palette, volume loading and display helpers for the paper's figures.

The three categorical colours are a colour-vision-deficiency-safe set that stays distinguishable
in every pair: blue and orange separate the two grades, and the enhancing / necrotic / edema
triad reuses the same three so a reader who has learned one figure can read the next.
"""
import numpy as np
import pandas as pd

from . import config
from .subregions import EPS, estimate_subregions  # noqa: F401  re-exported for the figures

BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
SURFACE = "#ffffff"

HIGH_GRADE_COLOR, LOW_GRADE_COLOR = BLUE, ORANGE

#: Enhancing is warm, the necrotic core takes the dark colour, edema the third.
SUB_COLORS = {"ET": ORANGE, "NCR": BLUE, "ED": AQUA}
SUB_LABELS = {"ET": "Enhancing (ET)", "NCR": "Necrotic core (NCR)", "ED": "Edema (ED)"}

#: Sequential, colour-vision-deficiency-safe, for the enhancement heat map.
ENHANCEMENT_CMAP = "cividis"

#: Agreement map colours: overlap, missed, predicted beyond the reference.
OVERLAP_COLOR, MISSED_COLOR, EXTRA_COLOR = AQUA, BLUE, ORANGE


def set_style():
    import matplotlib as mpl

    mpl.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 320,
        "savefig.bbox": "tight",
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
        "font.size": 9,
        "axes.titlesize": 9.5,
        "axes.labelsize": 9,
        "axes.edgecolor": MUTED,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.labelcolor": INK,
        "text.color": INK,
        "xtick.color": INK_SECONDARY,
        "ytick.color": INK_SECONDARY,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "legend.frameon": False,
        "grid.color": GRID,
        "grid.linewidth": 0.7,
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
    })


def grade_map():
    """Grades for labelling the figure rows."""
    from .cohort import load_grade_labels
    return load_grade_labels()


def load_case(case_id, seg_dir=None):
    """The four modalities, the predicted whole-tumor mask and, where present, the reference."""
    seg_dir = seg_dir or config.SEG_DIR
    data = np.load(str(config.NPZ_DIR / f"{case_id}.npz"))
    WT = np.load(str(seg_dir / f"{case_id}_seg.npy")).astype(bool)
    return {
        "flair": data["flair"].astype(np.float32),
        "t1": data["t1"].astype(np.float32),
        "t1ce": data["t1ce"].astype(np.float32),
        "t2": data["t2"].astype(np.float32),
        "wt": WT,
        # One case in the cohort carries no reference mask.
        "seg": data["seg"] if "seg" in data.files else None,
    }


def largest_slice(mask):
    """Index of the axial slice with the largest masked area."""
    return int(np.argmax(mask.reshape(mask.shape[0], -1).sum(axis=1)))


def crop_bbox(mask_2d, pad=18):
    """Row and column slices around a mask, padded and clipped."""
    ys, xs = np.where(mask_2d)
    y0, y1 = max(ys.min() - pad, 0), min(ys.max() + pad + 1, mask_2d.shape[0])
    x0, x1 = max(xs.min() - pad, 0), min(xs.max() + pad + 1, mask_2d.shape[1])
    return slice(y0, y1), slice(x0, x1)


def window(image, lo=1, hi=99):
    """Robust grayscale window to the unit interval, for display only."""
    a, b = np.percentile(image, [lo, hi])
    return np.clip((image - a) / (b - a + EPS), 0, 1)


def dice(predicted, reference):
    return 2 * float((predicted & reference).sum()) / (
        float(predicted.sum()) + float(reference.sum()) + EPS)


def paint(ax, gray, layers, alpha=0.55):
    """Grayscale background with translucent mask layers on top."""
    from matplotlib.colors import to_rgba

    ax.imshow(gray, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    for mask, color in layers:
        rgba = np.zeros((*mask.shape, 4))
        rgba[mask] = to_rgba(color, alpha=alpha)
        ax.imshow(rgba, interpolation="nearest")


def bare(ax):
    """Strip ticks and spines, since these panels are images and not plots."""
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def save(fig, name, out_dir=None):
    """Write a figure to the output directory and report where it went."""
    out_dir = out_dir or config.FIGURE_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    fig.savefig(path)
    print(f"wrote {path}")
    return path


def feature_table():
    """The seventeen features grouped by what they measure, for the method notebook."""
    from .subregions import FEATURE_GROUPS

    rows = [{"Feature": name, "Group": group}
            for group, names in FEATURE_GROUPS.items() for name in names]
    return pd.DataFrame(rows).set_index("Feature")
