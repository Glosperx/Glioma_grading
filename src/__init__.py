"""Shared code behind the notebooks: one implementation of the pipeline, one protocol.

    config        paths and pretrained-model identifiers, resolved from the environment
    subregions    label-free sub-region estimation and the seventeen volumetric features
    cohort        grades, feature extraction, the three cohorts, the encoder blocks
    evaluation    the cross-validation protocol, the classifier panels, fold-local PCA
    stats         paired bootstrap, error correlation, oracle upper bounds
    segmentation  Dice / IoU / precision / recall and ground-truth sub-region recovery
    medsam2       the whole-tumor fine-tune and inference
    encoders      the BiomedCLIP and MedGemma inputs, prompts and embeddings
    routing       the per-case routing experiment
    figures       palette and volume loading for the figures
    tables        the paper's tables, built from the modules above
"""
__all__ = [
    "config", "subregions", "cohort", "evaluation", "stats",
    "segmentation", "medsam2", "encoders", "routing", "figures", "tables",
]
