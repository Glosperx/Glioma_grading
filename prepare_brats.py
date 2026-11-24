"""Convert the BraTS2020 training data into the layout the notebooks read.

    python prepare_brats.py /path/to/MICCAI_BraTS2020_TrainingData

Writes, under GLIOMA_DATA_ROOT (default data/):

    brats_npz_all/{case}.npz   flair, t1, t1ce, t2 and seg, each of shape (155, 240, 240)
    brats_meta/name_mapping.csv

Each modality is clipped to its 1st and 99th percentiles and scaled to uint8.
"""
import argparse
import os
import shutil
from pathlib import Path

import nibabel as nib
import numpy as np
from tqdm import tqdm

MODALITIES = ["flair", "t1", "t1ce", "t2"]


def normalize(volume):
    lo, hi = np.percentile(volume, 1), np.percentile(volume, 99)
    return ((np.clip(volume, lo, hi) - lo) / (hi - lo + 1e-8) * 255).astype(np.uint8)


def load(path):
    # NIfTI volumes are (height, width, slices); the notebooks use (slices, height, width).
    return np.transpose(nib.load(str(path)).get_fdata(dtype=np.float32), (2, 0, 1))


def find(case_dir, suffix):
    path = case_dir / f"{case_dir.name}_{suffix}.nii"
    return path if path.exists() else case_dir / f"{case_dir.name}_{suffix}.nii.gz"


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("brats_dir", type=Path, help="the MICCAI_BraTS2020_TrainingData folder")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(os.environ.get("GLIOMA_DATA_ROOT", Path(__file__).parent / "data")),
    )
    args = parser.parse_args()

    npz_dir = args.out / "brats_npz_all"
    meta_dir = args.out / "brats_meta"
    npz_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)

    cases = sorted(d for d in args.brats_dir.iterdir() if d.is_dir())
    written, no_seg = 0, []
    for case_dir in tqdm(cases):
        paths = {m: find(case_dir, m) for m in MODALITIES}
        if not all(p.exists() for p in paths.values()):
            print(f"skipped {case_dir.name}: missing modality")
            continue
        arrays = {m: normalize(load(p)) for m, p in paths.items()}
        seg_path = find(case_dir, "seg")
        if seg_path.exists():
            arrays["seg"] = load(seg_path).astype(np.uint8)
        else:
            no_seg.append(case_dir.name)
        np.savez_compressed(npz_dir / f"{case_dir.name}.npz", **arrays)
        written += 1

    shutil.copy(args.brats_dir / "name_mapping.csv", meta_dir / "name_mapping.csv")
    print(f"{written} cases written to {npz_dir}")
    if no_seg:
        print("no segmentation file for:", ", ".join(no_seg))


if __name__ == "__main__":
    main()
