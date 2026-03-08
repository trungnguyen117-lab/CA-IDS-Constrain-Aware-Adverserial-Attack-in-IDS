"""Centralised path resolution for the foami+ project.

Directory layout assumed:
    SOICT25/                ← ROOT_DIR
    ├── src/                ← SRC_DIR   (art_classifier, art_generator, utils…)
    ├── models/             ← MODELS_DIR
    ├── DS/                 ← DATA_DIR
    ├── adv_samples/        ← ADV_DIR
    └── foami+/             ← FOAMI_DIR
        ├── utils/          ← this file lives here
        ├── training/
        ├── pipeline/
        └── report/         ← REPORT_DIR

Usage in a pipeline script
--------------------------
    import os, sys
    _HERE  = os.path.dirname(os.path.realpath(__file__))
    _FOAMI = os.path.dirname(os.path.dirname(_HERE))   # foami+/
    sys.path.insert(0, _FOAMI)

    from utils.paths import setup_paths, ROOT_DIR, FOAMI_DIR, MODELS_DIR, ...
    setup_paths()   # adds SRC_DIR + FOAMI_DIR to sys.path
"""
import os
import sys

# ── Derived from this file's own location ────────────────────────────────────
_UTILS_DIR = os.path.dirname(os.path.realpath(__file__))   # foami+/utils/

FOAMI_DIR   = os.path.dirname(_UTILS_DIR)                  # foami+/
ROOT_DIR    = os.path.dirname(FOAMI_DIR)                   # SOICT25/

SRC_DIR     = os.path.join(ROOT_DIR,  'src')
MODELS_DIR  = os.path.join(ROOT_DIR,  'models')
DATA_DIR    = os.path.join(ROOT_DIR,  'DS')
ADV_DIR     = os.path.join(ROOT_DIR,  'adv_samples')
REPORT_DIR  = os.path.join(FOAMI_DIR, 'report')

# Default dataset files
TRAIN_CSV      = os.path.join(DATA_DIR,  'train_tvae.csv')
TRAIN_ORIG_CSV = os.path.join(DATA_DIR,  'train_shap_66.csv')   # original (pre-TVAE)
TEST_CSV       = os.path.join(DATA_DIR,  'test_shap_66.csv')

# Adversarial-training data directory (adv generated from TVAE train data → used for AT)
AT_DIR        = os.path.join(ADV_DIR,  'adv_training')
AT_TRAIN_CSV  = os.path.join(AT_DIR,   'train_at.csv')        # TVAE-augmented
AT_MERGED_CSV = os.path.join(AT_DIR,   'train_at_merged.csv') # TVAE + adversarial examples

# Adversarial-evaluation directory (adv generated from test set → used ONLY for evaluation)
# Keeping this separate from AT_DIR avoids any form of data leakage between
# the adversarial-training pipeline and the benchmark evaluation pipeline.
ADV_EVAL_DIR  = os.path.join(ADV_DIR,  'adv_eval')

# Adaptive adversarial-evaluation directory (adv generated from test set using AT models)
# These examples are crafted against the AT-hardened models, providing a fair
# evaluation of adversarial robustness after adversarial training.
ADV_EVAL_AT_DIR = os.path.join(ADV_DIR, 'adv_eval_at')


def adv_csv(target: str, attack: str, adv_dir: str = ADV_DIR) -> str:
    """Return the default adversarial CSV path for a given target + attack."""
    return os.path.join(adv_dir, target, f"{target}_{attack}_adv.csv")


def setup_paths() -> None:
    """Insert SRC_DIR and FOAMI_DIR into sys.path (idempotent)."""
    for p in (SRC_DIR, FOAMI_DIR):
        if p not in sys.path:
            sys.path.insert(0, p)
