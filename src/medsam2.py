"""Stage one: whole-tumor segmentation with a fine-tuned MedSAM2 (paper Sections 3.1, 4.1).

MedSAM2 is promptable, so the fine-tune is slice by slice with a bounding box taken from the
reference mask. Only the mask decoder, the prompt encoder and the top of the image encoder
train; the pretrained backbone stays frozen. Per-slice masks are stacked back into a volume,
and that volume is the only definition of the tumor the later stages see.

Two inputs are compared, and the paper reports both as equivalent:

    flair                  a single channel replicated three times
    flair, t1ce, t2        one modality per channel

The channel set decides the checkpoint and output directory, so the two runs cannot overwrite
each other.

The masks every downstream table reads come from a five-fold cross-fit of the FLAIR input on the
grading folds themselves (``data_cache/folds.csv``): the masks of fold k are predicted by a model
fine-tuned on the other four folds for twelve epochs, and the final-epoch weights are kept with no
checkpoint selection. No case is therefore graded from a mask predicted by a model that saw it.
The three-channel input was compared on a single 80/20 case split and was not cross-fitted.

Volumes are stored already normalized -- clipped to their 1st and 99th percentiles and scaled
to 8-bit. ``normalization_proof`` shows this rather than assuming it, because applying the same
transform a second time would shift voxels and break comparability with the input the published
fine-tune saw.
"""
from . import _openmp  # noqa: F401  keep xgboost ahead of torch on macOS

import random
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from . import config

# Fine-tuning settings. Every one of these is part of the published recipe.
IMG_SIZE = 512                 # must match the MedSAM2 configuration file
SLICES_PER_CASE = 35           # tumor-bearing slices sampled uniformly along the axial axis
VAL_RATIO = 0.2                # split at case level, so no slice of a validation case is seen
BATCH_SIZE = 4
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 1.0
BBOX_MARGIN = 10               # pixels of slack around the reference bounding box
EPOCHS = 6                     # the single-split runs that compare the two inputs
CROSSFIT_EPOCHS = 12           # per fold of the cross-fit, final-epoch weights kept
VOLUME_SHAPE = (155, 240, 240)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

CHANNEL_SETS = {"flair": ["flair"], "multichannel": ["flair", "t1ce", "t2"]}
CROSSFIT_CHANNELS = CHANNEL_SETS["flair"]


def pick_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def run_paths(channels, data_root=None):
    """Checkpoint, mask directory and feature cache, keyed on the channel set."""
    root = data_root or config.DATA_ROOT
    tag = "mc" if len(channels) == 3 else channels[0]
    seg_dir = root / ("brats_seg_mc" if tag == "mc" else f"brats_seg_{tag}_local")
    return tag, root / f"medsam2_{tag}_best.pt", seg_dir, config.CACHE_DIR / f"features_{tag}17.npz"


def hyperparameters():
    """The recipe as a table, so a notebook can show it without restating it in prose."""
    rows = [
        ("checkpoint", f"{config.MEDSAM2_REPO} / {config.MEDSAM2_CHECKPOINT}"),
        ("configuration", config.MEDSAM2_CONFIG),
        ("input size", f"{IMG_SIZE} x {IMG_SIZE}"),
        ("trainable", "mask decoder, prompt encoder, image encoder neck"),
        ("frozen", "the pretrained backbone"),
        ("slices per case", f"{SLICES_PER_CASE}, tumor-bearing, uniform along the axial axis"),
        ("cross-fit", "the five grading folds (stratified, shuffled, seed 42); fold k is "
                      "segmented by a model fine-tuned on the other four"),
        ("cross-fit epochs", f"{CROSSFIT_EPOCHS} per fold, final-epoch weights, "
                             "no checkpoint selection, seed 42 + k"),
        ("input comparison split", f"case level, {int((1 - VAL_RATIO) * 100)}/"
                                   f"{int(VAL_RATIO * 100)}, seed 42, best validation epoch"),
        ("batch size", BATCH_SIZE),
        ("optimizer", f"AdamW, lr {LEARNING_RATE}, weight decay {WEIGHT_DECAY}"),
        ("schedule", "cosine annealing over the epochs"),
        ("gradient clipping", GRAD_CLIP),
        ("loss", "half Dice, half binary cross-entropy"),
        ("prompt", f"reference bounding box with a {BBOX_MARGIN} px margin"),
        ("augmentation", "random horizontal flip, box mirrored with it"),
        ("input comparison epochs", EPOCHS),
    ]
    return pd.DataFrame(rows, columns=["setting", "value"]).set_index("setting")


# ---------------------------------------------------------------------------
# Normalization already applied to the stored volumes
# ---------------------------------------------------------------------------
def normalize_volume(vol):
    """The transform the stored volumes have already been through, kept for the proof below."""
    lo, hi = np.percentile(vol, 1), np.percentile(vol, 99)
    vol = np.clip(vol, lo, hi)
    return ((vol - lo) / (hi - lo + 1e-8) * 255).astype(np.uint8)


def normalization_proof(channels, n_cases=5):
    """Show that re-applying the transform would only re-round, so the arrays are used as is."""
    rows = []
    for path in sorted(config.NPZ_DIR.glob("*.npz"))[:n_cases]:
        data = np.load(str(path))
        for channel in channels:
            vol = data[channel]
            again = normalize_volume(vol)
            rows.append({
                "case": path.stem, "channel": channel, "dtype": str(vol.dtype),
                "1st percentile": round(float(np.percentile(vol, 1)), 1),
                "99th percentile": round(float(np.percentile(vol, 99)), 1),
                "voxels changed by re-applying": round(float((vol != again).mean()), 3),
                "largest change": int(np.abs(vol.astype(int) - again.astype(int)).max()),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
class BraTSSliceDataset(Dataset):
    """Per-slice images, reference masks and box prompts, resized for the encoder."""

    def __init__(self, slices, img_size=IMG_SIZE, augment=False):
        self.slices = slices
        self.img_size = img_size
        self.augment = augment

    def __len__(self):
        return len(self.slices)

    def __getitem__(self, idx):
        img, gt, bbox, height, width = self.slices[idx]

        if self.augment and random.random() > 0.5:
            img = np.flip(img, axis=-1).copy()
            gt = np.fliplr(gt).copy()
            x0, y0, x1, y1 = bbox
            bbox = [width - x1, y0, width - x0, y1]

        img = img.astype(np.float32) / 255.0
        if img.shape[0] == 1:
            img = np.repeat(img, 3, axis=0)          # one modality, replicated three times
        img_t = torch.from_numpy(img).unsqueeze(0)
        img_r = F.interpolate(img_t, (self.img_size, self.img_size),
                              mode="bilinear", align_corners=False).squeeze(0)
        mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
        std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
        img_r = (img_r - mean) / std

        gt_t = torch.from_numpy(gt.astype(np.float32)).unsqueeze(0).unsqueeze(0)
        gt_r = F.interpolate(gt_t, (self.img_size // 4, self.img_size // 4),
                             mode="nearest").squeeze(0)

        scale_x, scale_y = self.img_size / width, self.img_size / height
        x0, y0, x1, y1 = bbox
        box = torch.tensor([x0 * scale_x, y0 * scale_y, x1 * scale_x, y1 * scale_y],
                           dtype=torch.float32)
        return img_r, gt_r, box


def bbox_from_mask(mask_2d, margin=BBOX_MARGIN):
    """Bounding box of a mask, padded and clipped. ``None`` when the slice holds no tumor."""
    ys, xs = np.where(mask_2d > 0)
    if len(ys) == 0:
        return None
    height, width = mask_2d.shape
    return [max(0, int(xs.min()) - margin), max(0, int(ys.min()) - margin),
            min(width, int(xs.max()) + margin), min(height, int(ys.max()) + margin)]


def sample_slices(npz_path, channels, max_slices=SLICES_PER_CASE):
    """Up to ``max_slices`` tumor-bearing slices, spread evenly along the axial axis."""
    data = np.load(str(npz_path), allow_pickle=True)
    if "seg" not in data.files:
        return []                                    # no reference mask, no supervision
    gts = (data["seg"] > 0).astype(np.uint8)
    volumes = [data[c] for c in channels]
    depth, height, width = gts.shape

    tumor_slices = [i for i in range(depth) if gts[i].sum() > 0]
    if not tumor_slices:
        return []
    if len(tumor_slices) > max_slices:
        step = len(tumor_slices) / max_slices
        chosen = [tumor_slices[int(i * step)] for i in range(max_slices)]
    else:
        chosen = tumor_slices

    out = []
    for i in chosen:
        gt = gts[i]
        img = np.stack([v[i] for v in volumes], 0)
        out.append((img, gt.copy(), bbox_from_mask(gt), height, width))
    return out


def case_split(val_ratio=VAL_RATIO, seed=42):
    """Case-level split. Splitting by slice would leak a case across the boundary."""
    all_npz = sorted(config.NPZ_DIR.glob("*.npz"))
    random.seed(seed)
    random.shuffle(all_npz)
    k = int(len(all_npz) * (1 - val_ratio))
    return all_npz[:k], all_npz[k:]


def preload(npz_files, channels, desc="Preload"):
    slices = []
    for path in tqdm(npz_files, desc=desc):
        slices.extend(sample_slices(path, channels))
    return slices


def build_rgb_slice(data, channels, i):
    """One slice as height-width-channel 8-bit, for the predictor.

    The predictor applies the same resize and normalization as training, so preprocessing
    matches by construction rather than by a second copy of the arithmetic.
    """
    if len(channels) == 1:
        return np.stack([data[channels[0]][i]] * 3, axis=-1)
    return np.stack([data[c][i] for c in channels], axis=-1)


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------
class DiceLoss(torch.nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        pred = torch.sigmoid(pred).view(-1)
        target = target.view(-1)
        intersection = (pred * target).sum()
        return 1 - (2 * intersection + self.smooth) / (pred.sum() + target.sum() + self.smooth)


class DiceBCELoss(torch.nn.Module):
    """Half Dice, half binary cross-entropy."""

    def __init__(self):
        super().__init__()
        self.dice = DiceLoss()
        self.bce = torch.nn.BCEWithLogitsLoss()

    def forward(self, pred, target):
        return 0.5 * self.dice(pred, target) + 0.5 * self.bce(pred, target)


def dice_coeff(pred_logits, target, threshold=0.5):
    """Batch-mean Dice at the slice level, which is what the training log reports."""
    pred = (torch.sigmoid(pred_logits) > threshold).float()
    batch = pred.shape[0]
    p, t = pred.view(batch, -1), target.view(batch, -1)
    intersection = (p * t).sum(1)
    return ((2 * intersection + 1e-6) / (p.sum(1) + t.sum(1) + 1e-6)).mean().item()


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
def build_model(mode="train", device=None):
    """Load MedSAM2 from its published checkpoint.

    On Metal the fused attention kernel selector is a CUDA-only context, so plain scaled
    dot-product attention is enabled instead.
    """
    device = str(device or pick_device())
    import sam2.modeling.sam.transformer as sam_transformer
    sam_transformer.ALLOW_ALL_KERNELS = True
    from huggingface_hub import hf_hub_download
    from sam2.build_sam import build_sam2

    checkpoint = hf_hub_download(repo_id=config.MEDSAM2_REPO,
                                 filename=config.MEDSAM2_CHECKPOINT)
    return build_sam2(config.MEDSAM2_CONFIG, checkpoint, device=device, mode=mode)


def freeze_and_optimise(model, epochs=EPOCHS):
    """Train the mask decoder, the prompt encoder and the encoder neck; freeze the rest."""
    for p in model.parameters():
        p.requires_grad = False
    for module in (model.sam_mask_decoder, model.sam_prompt_encoder, model.image_encoder.neck):
        for p in module.parameters():
            p.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"trainable {trainable:,} of {total:,} parameters "
          f"({100 * trainable / total:.1f}%)")

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    return optimizer, scheduler


def forward_pass(model, images, boxes):
    """Image encoder, box prompt, mask decoder: one low-resolution mask per slice."""
    backbone_out = model.forward_image(images)
    _, vision_feats, _, feat_sizes = model._prepare_backbone_features(backbone_out)
    if model.directly_add_no_mem_embed:
        vision_feats[-1] = vision_feats[-1] + model.no_mem_embed

    feats = [f.permute(1, 2, 0).view(images.shape[0], -1, *s)
             for f, s in zip(vision_feats[::-1], feat_sizes[::-1])][::-1]
    image_embedding, high_res_feats = feats[-1], feats[:-1]

    sparse, dense = model.sam_prompt_encoder(points=None, boxes=boxes.unsqueeze(1), masks=None)
    low_res_masks, _, _, _ = model.sam_mask_decoder(
        image_embeddings=image_embedding,
        image_pe=model.sam_prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse,
        dense_prompt_embeddings=dense,
        multimask_output=False,
        repeat_image=False,
        high_res_features=high_res_feats,
    )
    return low_res_masks


# ---------------------------------------------------------------------------
# Fine-tuning and inference
# ---------------------------------------------------------------------------
def fine_tune(channels, epochs=EPOCHS, device=None, checkpoint_path=None):
    """Fine-tune on one channel set, keeping the best epoch by validation Dice.

    Validation Dice is a slice-level number, unlike the volumetric Dice of tab:seg, so the two
    are not directly comparable.
    """
    device = pick_device() if device is None else device
    tag, default_ckpt, _, _ = run_paths(channels)
    checkpoint_path = checkpoint_path or default_ckpt
    print(f"channels {channels} | tag {tag} | device {device} | epochs {epochs}")

    train_files, val_files = case_split()
    train_loader = DataLoader(BraTSSliceDataset(preload(train_files, channels, "Preload train"),
                                               augment=True),
                              batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(BraTSSliceDataset(preload(val_files, channels, "Preload val"),
                                              augment=False),
                            batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    print(f"{len(train_loader)} training batches, {len(val_loader)} validation batches")

    model = build_model("train", device)
    optimizer, scheduler = freeze_and_optimise(model, epochs)
    criterion = DiceBCELoss().to(device)

    history, best = {"loss": [], "train Dice": [], "validation Dice": []}, 0.0
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = epoch_dice = 0.0
        for images, gts, boxes in tqdm(train_loader, desc=f"Train {epoch}/{epochs}"):
            images, gts, boxes = images.to(device), gts.to(device), boxes.to(device)
            optimizer.zero_grad()
            low_res = forward_pass(model, images, boxes)
            loss = criterion(low_res, gts)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], GRAD_CLIP)
            optimizer.step()
            epoch_loss += loss.item()
            epoch_dice += dice_coeff(low_res, gts)
        scheduler.step()

        model.eval()
        val_sum = 0.0
        with torch.no_grad():
            for images, gts, boxes in tqdm(val_loader, desc=f"Validate {epoch}/{epochs}"):
                images, gts, boxes = images.to(device), gts.to(device), boxes.to(device)
                val_sum += dice_coeff(forward_pass(model, images, boxes), gts)

        history["loss"].append(epoch_loss / len(train_loader))
        history["train Dice"].append(epoch_dice / len(train_loader))
        history["validation Dice"].append(val_sum / len(val_loader))

        note = ""
        if history["validation Dice"][-1] > best:
            best = history["validation Dice"][-1]
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), str(checkpoint_path))
            note = f"  best so far, saved to {checkpoint_path.name}"
        print(f"epoch {epoch:>2} | loss {history['loss'][-1]:.4f} | "
              f"train Dice {history['train Dice'][-1]:.4f} | "
              f"validation Dice {history['validation Dice'][-1]:.4f}{note}")

    print(f"\nbest validation Dice {best:.4f}")
    return pd.DataFrame(history, index=range(1, epochs + 1)), best


def segment_all(channels, checkpoint_path=None, seg_out=None, device=None, force=False,
                model=None):
    """Segment every case and stack the per-slice masks into a volume.

    ``model=None`` loads the fine-tuned checkpoint. Passing a freshly built model instead gives
    the zero-shot baseline, which is the pre-fine-tuning number the paper compares against.
    """
    device = pick_device() if device is None else device
    _, default_ckpt, default_out, _ = run_paths(channels)
    seg_out = seg_out or default_out
    seg_out.mkdir(parents=True, exist_ok=True)

    if model is None:
        checkpoint_path = checkpoint_path or default_ckpt
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"no checkpoint at {checkpoint_path}; fine-tune first")
        model = build_model("eval", device)
        model.load_state_dict(torch.load(str(checkpoint_path), map_location=str(device)))
    model.eval()

    from sam2.sam2_image_predictor import SAM2ImagePredictor
    predictor = SAM2ImagePredictor(model)

    for npz_path in tqdm(sorted(config.NPZ_DIR.glob("*.npz")), desc="Segmenting"):
        out_path = seg_out / f"{npz_path.stem}_seg.npy"
        if out_path.exists() and not force:
            continue
        np.save(str(out_path), predict_case(predictor, npz_path, channels))
    return seg_out


def predict_case(predictor, npz_path, channels):
    """One case, slice by slice, box-prompted, stacked back into a volume."""
    data = np.load(str(npz_path), allow_pickle=True)
    has_reference = "seg" in data.files
    references = (data["seg"] > 0) if has_reference else None
    depth, height, width = data[channels[0]].shape
    predicted = np.zeros((depth, height, width), dtype=np.uint8)
    for i in range(depth):
        # Without a reference mask there is no box to prompt with, so the whole frame is
        # used. Only the one case that carries no reference takes this path.
        box = (bbox_from_mask(references[i]) if has_reference else [0, 0, width, height])
        if box is None:
            continue
        predictor.set_image(build_rgb_slice(data, channels, i))
        with torch.no_grad():
            masks, _, _ = predictor.predict(box=np.array([box]), multimask_output=False)
        predicted[i] = masks[0].astype(np.uint8)
    return predicted


# ---------------------------------------------------------------------------
# Cross-fit: the masks the downstream tables read
# ---------------------------------------------------------------------------
def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def crossfit_files(k):
    """(training volumes, test volumes) of fold k, from the shipped fold file."""
    from .cohort import load_folds

    folds = load_folds()
    train = [config.NPZ_DIR / f"{c}.npz" for c in folds.case_id[folds.fold != k]]
    test = [config.NPZ_DIR / f"{c}.npz" for c in folds.case_id[folds.fold == k]]
    assert not {p.stem for p in train} & {p.stem for p in test}
    return train, test


def crossfit_checkpoint(k, data_root=None):
    return (data_root or config.DATA_ROOT) / f"medsam2_crossfit_fold{k}.pt"


def slice_dice(model, npz_files, channels, device):
    """Mean slice-level Dice at decoder resolution, the same measure as the validation Dice."""
    loader = DataLoader(BraTSSliceDataset(preload(npz_files, channels, "Preload test fold"),
                                          augment=False),
                        batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    model.eval()
    total = 0.0
    with torch.no_grad():
        for images, gts, boxes in tqdm(loader, desc="Test-fold slice Dice"):
            images, gts, boxes = images.to(device), gts.to(device), boxes.to(device)
            total += dice_coeff(forward_pass(model, images, boxes), gts)
    return total / len(loader), len(loader.dataset)


def fine_tune_fold(k, epochs=CROSSFIT_EPOCHS, channels=CROSSFIT_CHANNELS, device=None,
                   checkpoint_path=None):
    """Fine-tune on the four folds other than k and keep the final-epoch weights.

    There is no validation split and no checkpoint selection, so nothing about fold k is seen
    before its masks are predicted. Fold k is scored once afterwards at slice level, for
    reporting only; that is the per-slice Dice the paper quotes as 0.86.
    """
    device = pick_device() if device is None else device
    checkpoint_path = checkpoint_path or crossfit_checkpoint(k)
    train_files, test_files = crossfit_files(k)
    set_seeds(42 + k)
    print(f"fold {k} | {len(train_files)} training cases, {len(test_files)} test cases | "
          f"channels {channels} | epochs {epochs} | device {device}")

    loader = DataLoader(BraTSSliceDataset(preload(train_files, channels, f"Preload fold {k}"),
                                          augment=True),
                        batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    model = build_model("train", device)
    optimizer, scheduler = freeze_and_optimise(model, epochs)
    criterion = DiceBCELoss().to(device)

    history = {"loss": [], "train Dice": [], "minutes": []}
    for epoch in range(1, epochs + 1):
        started = time.time()
        model.train()
        epoch_loss = epoch_dice = 0.0
        for images, gts, boxes in tqdm(loader, desc=f"Fold {k} train {epoch}/{epochs}"):
            images, gts, boxes = images.to(device), gts.to(device), boxes.to(device)
            optimizer.zero_grad()
            low_res = forward_pass(model, images, boxes)
            loss = criterion(low_res, gts)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], GRAD_CLIP)
            optimizer.step()
            epoch_loss += loss.item()
            epoch_dice += dice_coeff(low_res, gts)
        scheduler.step()
        history["loss"].append(epoch_loss / len(loader))
        history["train Dice"].append(epoch_dice / len(loader))
        history["minutes"].append((time.time() - started) / 60)
        print(f"fold {k} epoch {epoch:>2} | loss {history['loss'][-1]:.4f} | "
              f"train Dice {history['train Dice'][-1]:.4f}")

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), str(checkpoint_path))
    test_dice, n_slices = slice_dice(model, test_files, channels, device)
    print(f"fold {k}: final-epoch weights saved; test-fold slice Dice {test_dice:.4f} "
          f"over {n_slices} slices")
    return pd.DataFrame(history, index=range(1, epochs + 1)), test_dice


def segment_fold(k, channels=CROSSFIT_CHANNELS, checkpoint_path=None, seg_out=None,
                 device=None, force=False):
    """Masks for the cases of fold k only, from the model that did not see them.

    A mask already on disk is kept unless ``force=True``, so a run can resume. Re-running the
    cross-fit into a directory that already holds masks would therefore keep the old ones.
    """
    device = pick_device() if device is None else device
    checkpoint_path = checkpoint_path or crossfit_checkpoint(k)
    seg_out = seg_out or config.SEG_DIR
    seg_out.mkdir(parents=True, exist_ok=True)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"no checkpoint at {checkpoint_path}; run fine_tune_fold({k})")
    model = build_model("eval", device)
    model.load_state_dict(torch.load(str(checkpoint_path), map_location=str(device)))
    model.eval()

    from sam2.sam2_image_predictor import SAM2ImagePredictor
    predictor = SAM2ImagePredictor(model)
    _, test_files = crossfit_files(k)
    for npz_path in tqdm(test_files, desc=f"Segmenting fold {k}"):
        out_path = seg_out / f"{npz_path.stem}_seg.npy"
        if out_path.exists() and not force:
            continue
        np.save(str(out_path), predict_case(predictor, npz_path, channels))
    return seg_out


def crossfit_summary():
    """The shipped record of the five fine-tunes, with the mean the paper quotes."""
    table = pd.read_csv(config.CROSSFIT_TRAINING).set_index("fold")
    table.loc["mean"] = table.mean()
    return table.round({"test_slice_dice": 4, "final_train_dice": 4, "train_minutes": 1})


def check_masks(seg_out, expected_shape=VOLUME_SHAPE):
    """Every case present, right shape, right type, non-empty. Run before reporting numbers."""
    problems, empty, n = [], [], 0
    for npz_path in tqdm(sorted(config.NPZ_DIR.glob("*.npz")), desc="Checking masks"):
        n += 1
        mask_path = seg_out / f"{npz_path.stem}_seg.npy"
        if not mask_path.exists():
            problems.append(f"{npz_path.stem}: missing")
            continue
        mask = np.load(str(mask_path))
        if mask.shape != expected_shape:
            problems.append(f"{npz_path.stem}: shape {mask.shape}")
        if mask.dtype != np.uint8:
            problems.append(f"{npz_path.stem}: dtype {mask.dtype}")
        if mask.sum() == 0:
            empty.append(npz_path.stem)
    print(f"{n} cases | problems {len(problems)} | empty masks {len(empty)}")
    for line in problems[:10]:
        print("  " + line)
    return not problems and not empty


def timed_batches(channels, n_warmup=3, n_timed=20, cases=12, device=None):
    """Measure seconds per training iteration, so a run can be planned before it starts.

    The first calls on Metal pay kernel compilation, so they are discarded.
    """
    device = pick_device() if device is None else device
    train_files, _ = case_split()
    loader = DataLoader(BraTSSliceDataset(preload(train_files[:cases], channels, "Preload"),
                                          augment=True),
                        batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    model = build_model("train", device)
    optimizer, _ = freeze_and_optimise(model)
    criterion = DiceBCELoss().to(device)

    def sync():
        if device.type == "mps":
            torch.mps.synchronize()
        elif device.type == "cuda":
            torch.cuda.synchronize()

    model.train()
    times, seen = [], 0
    for images, gts, boxes in loader:
        images, gts, boxes = images.to(device), gts.to(device), boxes.to(device)
        sync()
        started = time.perf_counter()
        optimizer.zero_grad()
        low_res = forward_pass(model, images, boxes)
        loss = criterion(low_res, gts)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], GRAD_CLIP)
        optimizer.step()
        sync()
        seen += 1
        if seen > n_warmup:
            times.append(time.perf_counter() - started)
        if seen >= n_warmup + n_timed:
            break
    return {"seconds per iteration": round(float(np.mean(times)), 3),
            "median": round(float(np.median(times)), 3),
            "iterations timed": len(times)}
