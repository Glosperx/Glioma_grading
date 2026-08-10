"""The foundation-model branch: what each encoder is shown, and what comes back.

Both representations are computed from the predicted mask only, so the branch stays automatic
(paper Section 3.4). Both read one axial slice per case, the one with the largest predicted
tumor area.

MedGemma is shown two panels, the T1ce crop and the enhancement map of the same region, and
answers six categorical fields; the last-layer hidden state, mean-pooled over the answer
tokens, is the description embedding. BiomedCLIP is shown a single crop and its image
embedding is taken directly.

Both panels are windowed against healthy brain rather than against each crop's own percentiles.
Per-crop normalization was actively harmful: it stretches a non-enhancing tumor's noise across
the full range, so a low-grade and a high-grade case end up equally bright and the enhancement
panel stops carrying the signal it exists for. Fixed windowing in standard deviations makes
brightness mean the same thing in every case.
"""
from . import _openmp  # noqa: F401  keep xgboost ahead of torch on macOS

import csv
import gc
import re
import time

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm.auto import tqdm

from . import config
from .subregions import healthy_reference

# ---------------------------------------------------------------------------
# Region of interest, shared by both encoders
# ---------------------------------------------------------------------------
ROI_MARGIN = 8          # pixels of context around the predicted bounding box
PANEL_SIZE = 448        # each MedGemma panel; the two sit side by side
MAX_NEW_TOKENS = 96     # the questionnaire answer is short by construction

#: Display windows in standard deviations of healthy tissue.
ENHANCEMENT_WINDOW = (-2.0, 6.0)
T1CE_WINDOW = (-2.0, 4.0)


def largest_predicted_slice(WT):
    """Index of the slice holding the most predicted tumor. Never reads the reference mask."""
    areas = WT.reshape(WT.shape[0], -1).sum(axis=1)
    if areas.max() == 0:
        return WT.shape[0] // 2
    return int(np.argmax(areas))


def crop_roi(image_2d, mask_2d, margin=ROI_MARGIN):
    """Crop around the predicted bounding box, padded and clipped to the image."""
    if not mask_2d.any():
        return image_2d
    rows = np.where(mask_2d.any(axis=1))[0]
    cols = np.where(mask_2d.any(axis=0))[0]
    r0 = max(int(rows[0]) - margin, 0)
    r1 = min(int(rows[-1]) + margin + 1, image_2d.shape[0])
    c0 = max(int(cols[0]) - margin, 0)
    c1 = min(int(cols[-1]) + margin + 1, image_2d.shape[1])
    return image_2d[r0:r1, c0:c1]


def to_rgb_percentile(patch):
    """Percentile-normalize a patch to 8-bit and replicate to three channels."""
    patch = patch.astype(np.float32)
    lo, hi = np.percentile(patch, 1), np.percentile(patch, 99)
    scaled = np.clip((patch - lo) / (hi - lo + 1e-8), 0, 1)
    gray = (scaled * 255).astype(np.uint8)
    return Image.fromarray(np.stack([gray] * 3, axis=-1))


def to_rgb_windowed(patch, centre, spread, window):
    """Map a fixed number of healthy-tissue standard deviations to 8-bit."""
    sigmas = (patch.astype(np.float32) - centre) / (spread + 1e-6)
    low, high = window
    scaled = np.clip((sigmas - low) / (high - low), 0, 1)
    gray = (scaled * 255).astype(np.uint8)
    return Image.fromarray(np.stack([gray] * 3, axis=-1))


# ---------------------------------------------------------------------------
# BiomedCLIP
# ---------------------------------------------------------------------------
#: The two inputs the paper compares. Both are the same slice and the same crop; only the
#: image changes, so the comparison isolates what the encoder is shown.
BIOMEDCLIP_INPUTS = {
    "T1ce": "the T1ce crop",
    "enhancement": "the enhancement map T1ce - T1, cropped from the same region",
}


def build_biomedclip_inputs(t1ce, t1, WT):
    """Both compared inputs for one case, on the largest predicted slice."""
    idx = largest_predicted_slice(WT)
    mask = WT[idx]
    enhancement = np.clip(t1ce.astype(np.float32) - t1.astype(np.float32), 0, None)
    return {
        "T1ce": to_rgb_percentile(crop_roi(t1ce[idx], mask)),
        "enhancement": to_rgb_percentile(crop_roi(enhancement[idx], mask)),
    }, idx


def load_biomedclip(device=None):
    """Load BiomedCLIP and its preprocessing transform."""
    from open_clip import create_model_from_pretrained

    device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
    model, preprocess = create_model_from_pretrained(config.BIOMEDCLIP_ID)
    return model.to(device).eval(), preprocess, device


def embed_images(images, model, preprocess, device, batch_size=32):
    """Encode images into L2-normalized BiomedCLIP image embeddings."""
    vectors = []
    for start in range(0, len(images), batch_size):
        batch = images[start:start + batch_size]
        tensor = torch.stack([preprocess(img) for img in batch]).to(device)
        with torch.no_grad():
            feats = model.encode_image(tensor)
            feats = feats / feats.norm(dim=-1, keepdim=True)
        vectors.append(feats.float().cpu().numpy())
    return np.concatenate(vectors, axis=0)


def embed_biomedclip_all(which="T1ce", out_path=None, seg_dir=None, force=False):
    """Encode every case with one of the two inputs and cache the result.

    Needs the BraTS working copy. The two caches this produces ship with the repository, so the
    fusion notebook runs without repeating this step.
    """
    out_path = out_path or (config.BIOMEDCLIP_T1CE if which == "T1ce" else config.BIOMEDCLIP_ENH)
    if out_path.exists() and not force:
        print(f"already cached: {out_path.name}")
        return out_path
    seg_dir = seg_dir or config.SEG_DIR

    model, preprocess, device = load_biomedclip()
    seg_by_case = {p.stem.replace("_seg", ""): p for p in sorted(seg_dir.glob("*.npy"))}

    case_ids, images = [], []
    for npz_path in tqdm(sorted(config.NPZ_DIR.glob("*.npz")), desc=f"Regions ({which})"):
        case_id = npz_path.stem
        if case_id not in seg_by_case:
            continue
        data = np.load(str(npz_path), allow_pickle=True)
        t1ce = data["t1ce"].astype(np.float32)
        t1 = data["t1"].astype(np.float32)
        WT = np.load(str(seg_by_case[case_id])).astype(bool)
        if WT.shape != t1ce.shape:
            continue
        built, _ = build_biomedclip_inputs(t1ce, t1, WT)
        case_ids.append(case_id)
        images.append(built[which])

    embeddings = embed_images(images, model, preprocess, device)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, embeddings=embeddings, case_ids=np.array(case_ids))
    print(f"cached {embeddings.shape} to {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# MedGemma: the panels
# ---------------------------------------------------------------------------
def build_panels(t1ce, t1, flair, WT, panel_size=PANEL_SIZE):
    """The two-panel image MedGemma reads: T1ce crop beside the enhancement map.

    A four-panel montage was tried and abandoned: at 896 by 896 it crashes the Metal backend.
    The two-panel geometry is what the published results use.
    """
    idx = largest_predicted_slice(WT)
    enhancement = t1ce.astype(np.float32) - t1.astype(np.float32)

    healthy = healthy_reference(WT, flair)
    enh_mean, enh_std = float(enhancement[healthy].mean()), float(enhancement[healthy].std())
    t1ce_mean, t1ce_std = float(t1ce[healthy].mean()), float(t1ce[healthy].std())

    left = to_rgb_windowed(crop_roi(t1ce[idx], WT[idx]), t1ce_mean, t1ce_std, T1CE_WINDOW)
    right = to_rgb_windowed(crop_roi(enhancement[idx], WT[idx]), enh_mean, enh_std,
                            ENHANCEMENT_WINDOW)
    left = left.resize((panel_size, panel_size), Image.LANCZOS)
    right = right.resize((panel_size, panel_size), Image.LANCZOS)

    combined = Image.new("RGB", (panel_size * 2, panel_size))
    combined.paste(left, (0, 0))
    combined.paste(right, (panel_size, 0))
    return combined, idx


# ---------------------------------------------------------------------------
# MedGemma: the four prompt formulations
# ---------------------------------------------------------------------------
PANEL_DESCRIPTION = (
    "You are an expert neuroradiologist grading a glioma on brain MRI. "
    "The LEFT panel is an axial T1 post-contrast (T1ce) crop centred on the tumour. "
    "The RIGHT panel is the T1ce minus T1 enhancement map for the same region, windowed so that "
    "mid-grey is normal brain and bright means true contrast uptake above normal tissue. "
)

CATEGORICAL_FIELDS = (
    "Enhancement: <none|mild|avid>\n"
    "Necrosis: <absent|present>\n"
    "Margin: <well-defined|infiltrative>\n"
    "MassEffect: <none|mild|marked>\n"
    "EnhancementMapSignal: <low|intermediate|high>\n"
    "Assessment: <low-grade|high-grade>"
)

NUMERIC_FIELDS = (
    "EnhancingPercent: <0-100>\n"
    "NecroticPercent: <0-100>\n"
    "EdemaPercent: <0-100>\n"
    "MarginClarity: <0-100>\n"
    "MassEffect: <0-100>\n"
    "HighGradeConfidence: <0-100>"
)

# The calibration is written text with no image exemplars, so no case label can leak into the
# features and the cohort stays at 369. It does two things: it defines what each enhancement
# level looks like on the right-hand panel, and it counters the high-grade bias by saying that
# many gliomas do not enhance and that "none" is an expected answer.
FEWSHOT_CALIBRATION = (
    "Calibration, how to read these panels:\n"
    "- Avid enhancement: a clearly brighter-than-background rim or nodule on the RIGHT panel, "
    "usually ring-shaped around a dark centre. Typical of high-grade tumours.\n"
    "- Mild enhancement: patchy areas only slightly brighter than normal brain.\n"
    "- No enhancement: the tumour region on the RIGHT panel stays at or below the grey level of "
    "normal brain. This is common and is typical of low-grade tumours. Many gliomas do NOT "
    "enhance at all, so 'none' is a frequent and expected answer.\n"
    "- Necrosis: a dark core on the LEFT panel surrounded by enhancing tissue.\n"
    "Judge only what you can see. Do not assume a tumour is high-grade by default.\n"
)

FINAL_MARKER = "FINAL ANSWER:"

#: The four formulations of tab:prompts. ``marker`` is set when the model reasons first, so that
#: only the tokens after it are pooled and the vector represents the conclusion.
PROMPTS = {
    "baseline": {
        "label": "Baseline (categorical)",
        "system": (
            "You are an expert neuroradiologist grading a glioma on brain MRI. "
            "The LEFT panel is an axial T1 post-contrast (T1ce) crop centred on the tumour. "
            "The RIGHT panel is the T1ce minus T1 enhancement map for the same region, where "
            "bright means true contrast uptake. "
            "Answer ONLY with the following six fields, one per line, using exactly the allowed "
            "values and no other text:\n" + CATEGORICAL_FIELDS),
        "user": "Grade this glioma using the six fields.",
        "marker": None,
    },
    "numeric": {
        "label": "Numeric (percentages)",
        "system": (PANEL_DESCRIPTION
                   + "Answer ONLY with these six fields, one per line, each an integer 0-100 "
                     "and nothing else:\n" + NUMERIC_FIELDS),
        "user": "Estimate the six values for this glioma.",
        "marker": None,
    },
    "fewshot": {
        "label": "Few-shot",
        "system": (PANEL_DESCRIPTION + FEWSHOT_CALIBRATION
                   + "Answer ONLY with these six fields, one per line, using exactly the allowed "
                     "values and no other text:\n" + CATEGORICAL_FIELDS),
        "user": "Grade this glioma using the six fields.",
        "marker": None,
    },
    "fewshot_cot": {
        "label": "Few-shot $+$ reasoning",
        "system": (PANEL_DESCRIPTION + FEWSHOT_CALIBRATION
                   + "First reason briefly, in at most three sentences, about what you actually "
                     "see in each panel. Then write the line '" + FINAL_MARKER + "' on its own, "
                     "followed by these six fields, one per line, using exactly the allowed "
                     "values:\n" + CATEGORICAL_FIELDS),
        "user": "Examine the panels, reason briefly, then give the six fields.",
        "marker": FINAL_MARKER,
    },
}


def load_medgemma(device=None):
    """Load MedGemma and its processor."""
    from transformers import AutoModelForImageTextToText, AutoProcessor

    device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
    model = AutoModelForImageTextToText.from_pretrained(
        config.MEDGEMMA_ID, dtype=torch.bfloat16).to(device).eval()
    return model, AutoProcessor.from_pretrained(config.MEDGEMMA_ID), device


def ask_medgemma(image, model, processor, prompt, max_new_tokens=MAX_NEW_TOKENS):
    """Generate the answer, then mean-pool the last hidden layer over the answer tokens.

    Pooling the trailing positions rather than a fixed range is what makes this robust to how
    many tokens the image expands into in the prefix. When the prompt asks for reasoning first,
    only the tokens after the marker are pooled.
    """
    messages = [
        {"role": "system", "content": [{"type": "text", "text": prompt["system"]}]},
        {"role": "user", "content": [{"type": "image", "image": image},
                                     {"type": "text", "text": prompt["user"]}]},
    ]
    inputs = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt",
    ).to(model.device, dtype=torch.bfloat16)

    prompt_len = inputs["input_ids"].shape[-1]
    with torch.inference_mode():
        generated = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)

    answer_ids = generated[0][prompt_len:]
    answer = processor.decode(answer_ids, skip_special_tokens=True).strip()
    if answer_ids.shape[-1] == 0:
        return answer, None

    n_pool = int(answer_ids.shape[-1])
    if prompt["marker"] and prompt["marker"] in answer:
        final = answer.split(prompt["marker"])[-1].strip()
        n_pool = min(max(len(processor.tokenizer(final)["input_ids"]), 1), n_pool)

    forward_inputs = dict(inputs)
    forward_inputs["input_ids"] = generated
    forward_inputs["attention_mask"] = torch.ones_like(generated)
    with torch.inference_mode():
        outputs = model(**forward_inputs, output_hidden_states=True)

    pooled = outputs.hidden_states[-1][0][-n_pool:].float().mean(dim=0)
    return answer, pooled.cpu().numpy()


def iter_cases(seg_dir=None, limit=None):
    """Yield the volumes and predicted mask of each case that has both."""
    seg_dir = seg_dir or config.SEG_DIR
    npz_by_case = {p.stem: p for p in sorted(config.NPZ_DIR.glob("*.npz"))}
    seg_by_case = {p.stem.replace("_seg", ""): p for p in sorted(seg_dir.glob("*.npy"))}
    case_ids = sorted(c for c in npz_by_case if c in seg_by_case)
    for case_id in (case_ids[:limit] if limit else case_ids):
        data = np.load(str(npz_by_case[case_id]), allow_pickle=True)
        t1ce = data["t1ce"].astype(np.float32)
        t1 = data["t1"].astype(np.float32)
        flair = data["flair"].astype(np.float32)
        WT = np.load(str(seg_by_case[case_id])).astype(bool)
        if WT.shape != t1ce.shape:
            continue
        yield case_id, t1ce, t1, flair, WT


def _release(device):
    """Metal holds on to intermediates, and a long loop dies without this."""
    gc.collect()
    if str(device) == "mps":
        torch.mps.empty_cache()


def _save_hidden(path, by_case):
    if not by_case:
        return
    case_ids = sorted(by_case)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, hidden=np.stack([by_case[c] for c in case_ids]),
             case_ids=np.array(case_ids))


def run_medgemma(which="fewshot", limit=None, answers_path=None, hidden_path=None):
    """Run one prompt over the cohort, saving answers and hidden states as it goes.

    Resumable: answers already in the CSV are skipped, so an interrupted run loses nothing.
    Needs the BraTS working copy and the MedGemma weights; the resulting caches ship with the
    repository, so the tables can be rebuilt without this step.
    """
    prompt = PROMPTS[which]
    answers_path = answers_path or config.MEDGEMMA_ANSWERS[which]
    hidden_path = hidden_path or config.MEDGEMMA_HIDDEN[which]
    max_new = MAX_NEW_TOKENS * 3 if prompt["marker"] else MAX_NEW_TOKENS

    done = set()
    if answers_path.exists():
        with open(answers_path, newline="") as handle:
            done = {row["case_id"] for row in csv.DictReader(handle)}
    by_case = {}
    if hidden_path.exists():
        cached = np.load(hidden_path, allow_pickle=True)
        by_case = {str(c): v for c, v in zip(cached["case_ids"], cached["hidden"])}

    model, processor, device = load_medgemma()
    cases = list(iter_cases(limit=limit))
    print(f"prompt '{which}' | {len(cases)} cases | {len(done)} already answered")

    answers_path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not answers_path.exists()
    durations = []
    with open(answers_path, "a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["case_id", "slice_idx", "answer"])
        if is_new:
            writer.writeheader()
        for case_id, t1ce, t1, flair, WT in tqdm(cases, desc=which):
            if case_id in done:
                continue
            image, idx = build_panels(t1ce, t1, flair, WT)
            started = time.time()
            answer, pooled = ask_medgemma(image, model, processor, prompt, max_new)
            durations.append(time.time() - started)
            writer.writerow({"case_id": case_id, "slice_idx": idx, "answer": answer})
            handle.flush()
            if pooled is not None:
                by_case[case_id] = pooled
            if len(by_case) % 25 == 0:
                _save_hidden(hidden_path, by_case)
            _release(device)

    _save_hidden(hidden_path, by_case)
    if durations:
        print(f"{np.mean(durations):.1f} s per case over {len(durations)} new cases")
    return hidden_path


# ---------------------------------------------------------------------------
# MedGemma: what the answers themselves say
# ---------------------------------------------------------------------------
def parse_answers(path):
    """The saved answers as a table, one column per field.

    The model often echoes the template's angle brackets, so "<none>" and "none" arrive as
    different strings for the same answer and have to be normalized before anything is counted.
    """
    frame = pd.read_csv(path)
    parsed = []
    for answer in frame["answer"].astype(str):
        fields = {}
        for line in answer.splitlines():
            if ": " not in line:
                continue
            key, value = line.split(": ", 1)
            fields[key.strip()] = value.strip().strip("<>").strip()
        parsed.append(fields)
    return pd.DataFrame(parsed, index=frame["case_id"])


def answer_distribution(path, field="Enhancement", top=4):
    """How often each value of one field is answered, which is how saturation shows up."""
    table = parse_answers(path)
    if field not in table.columns:
        return pd.Series(dtype=int)
    return table[field].value_counts().head(top)


# ---------------------------------------------------------------------------
# MedGemma: its own confidence in the grade it wrote
# ---------------------------------------------------------------------------
ASSESSMENT_MARKER = "Assessment:"


def grade_token_ids(tokenizer):
    """First-token identifiers of the low and high branch, with and without a leading space."""
    def first(text):
        ids = tokenizer.encode(text, add_special_tokens=False)
        return ids[0] if ids else None

    low = {first(s) for s in (" low-grade", "low-grade", " low", "low")} - {None}
    high = {first(s) for s in (" high-grade", "high-grade", " high", "high")} - {None}
    return sorted(low), sorted(high)


def score_grade_token(image, answer, model, processor, low_ids, high_ids, prompt=None):
    """Recover the probability MedGemma decided the grade with, in one forward pass.

    The saved answer is teacher-forced up to and including the assessment field and the
    next-token distribution is read. Because the prefix is exactly what greedy decoding saw,
    the softmax at that position is the distribution the model actually chose from, rather than
    a classifier fitted on top of it.
    """
    prompt = prompt or PROMPTS["baseline"]
    idx = answer.find(ASSESSMENT_MARKER)
    if idx < 0:
        return None, None
    prefix = answer[: idx + len(ASSESSMENT_MARKER)]

    messages = [
        {"role": "system", "content": [{"type": "text", "text": prompt["system"]}]},
        {"role": "user", "content": [{"type": "image", "image": image},
                                     {"type": "text", "text": prompt["user"]}]},
        {"role": "assistant", "content": [{"type": "text", "text": prefix}]},
    ]
    # continue_final_message renders the assistant turn without its end token, so the next
    # predicted token is the grade word itself.
    inputs = processor.apply_chat_template(
        messages, add_generation_prompt=False, continue_final_message=True,
        tokenize=True, return_dict=True, return_tensors="pt",
    ).to(model.device, dtype=torch.bfloat16)

    with torch.inference_mode():
        logits = model(**inputs).logits[0, -1].float()
    probs = torch.softmax(logits, dim=-1)
    p_low, p_high = float(probs[low_ids].sum()), float(probs[high_ids].sum())
    if p_low + p_high == 0:
        return None, None
    return p_low / (p_low + p_high), ("low-grade" if p_low >= p_high else "high-grade")


def run_native_confidence(which="baseline", limit=None, out_path=None):
    """Extract that probability for every case whose answer is on disk.

    A sanity check comes free: taking the more likely of the two branches must reproduce the
    grade the saved answer states.
    """
    out_path = out_path or config.MEDGEMMA_NATIVE_CONF
    prompt = PROMPTS[which]

    with open(config.MEDGEMMA_ANSWERS[which], newline="") as handle:
        rows = list(csv.DictReader(handle))
    answers = {r["case_id"]: r["answer"] for r in rows}
    stated = {}
    for r in rows:
        found = re.search(r"Assessment:\s*(low-grade|high-grade)", r["answer"], re.IGNORECASE)
        stated[r["case_id"]] = found.group(1).lower() if found else None

    model, processor, device = load_medgemma()
    low_ids, high_ids = grade_token_ids(processor.tokenizer)

    collected, agree = [], 0
    for case_id, t1ce, t1, flair, WT in tqdm(list(iter_cases(limit=limit)), desc="Confidence"):
        answer = answers.get(case_id)
        if answer is None:
            continue
        image, _ = build_panels(t1ce, t1, flair, WT)
        p_low_grade, argmax_grade = score_grade_token(
            image, answer, model, processor, low_ids, high_ids, prompt)
        _release(device)
        if p_low_grade is None:
            continue
        matches = int(argmax_grade == stated.get(case_id))
        agree += matches
        collected.append((case_id, p_low_grade, stated.get(case_id), matches))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path,
             case_ids=np.array([r[0] for r in collected]),
             p_lgg_native=np.array([r[1] for r in collected], dtype=np.float32),
             hard_grade=np.array([r[2] for r in collected]),
             argmax_ok=np.array([r[3] for r in collected], dtype=np.int8))
    print(f"{len(collected)} cases saved to {out_path}")
    print(f"the more likely branch reproduces the stated grade in {agree}/{len(collected)} cases")
    return out_path
