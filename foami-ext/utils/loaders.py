"""Model loading helpers shared by generate and evaluate pipeline scripts.

All wrappers return ART-compatible estimator objects ready for adversarial
generation or evaluation.
"""
import importlib
import json
import os
import logging

import joblib
import numpy as np
import pandas as pd

from .constants import (
    GBT_TARGETS,
    DEFAULT_ENSEMBLE_WEIGHTS, DEFAULT_MI_W_GBT_BASE, DEFAULT_MI_PARAMS,
    LABEL_COL, MODEL_FILENAMES,
)
logger = logging.getLogger(__name__)


# ── File guard ────────────────────────────────────────────────────────────────

def require_file(path: str) -> None:
    """Raise SystemExit if *path* does not exist."""
    if not os.path.exists(path):
        raise SystemExit(f"File not found: {path}")


# ── ModelLoader class ─────────────────────────────────────────────────────────

class ModelLoader:
    """Registry-based model loader for ART-compatible wrappers.

    Usage:
        loader = ModelLoader(models_dir, clip_values, num_classes, input_dim, device)
        wrapper = loader.load('lstm')
        wrapper = loader.load('lstm', params={'gaussian_augmentation': True})
        wrappers = loader.load_multiple(['cat', 'rf', 'lstm', 'resdnn'])
    """

    _REGISTRY = {
        'xgb':    ('art_classifier.xgb_classifier',    'XGBWrapper'),
        'cat':    ('art_classifier.catb_classifier',    'CatBoostWrapper'),
        'rf':     ('art_classifier.sklearn_classifier', 'SkleanWrapper'),
        'lstm':   ('art_classifier.lstm_classifier',    'LSTMWrapper'),
        'resdnn': ('art_classifier.resdnn_classifier',  'ResDNNWrapper'),
    }

    def __init__(self, models_dir, clip_values, num_classes, input_dim, device):
        self.models_dir = models_dir
        self.clip_values = clip_values
        self.num_classes = num_classes
        self.input_dim = input_dim
        self.device = device

    def load(self, target, params=None):
        """Load model wrapper by target name."""
        if target not in self._REGISTRY:
            raise ValueError(f"Unknown target: {target}")
        module_path, class_name = self._REGISTRY[target]
        mod = importlib.import_module(module_path)
        wrapper_cls = getattr(mod, class_name)
        p = os.path.join(self.models_dir, MODEL_FILENAMES[target])
        require_file(p)
        if target in GBT_TARGETS:
            model = joblib.load(p)
            return wrapper_cls(model=model, num_classes=self.num_classes,
                               input_shape=(self.input_dim,),
                               clip_values=self.clip_values,
                               device=self.device, params=params)
        wrapper = wrapper_cls.from_checkpoint(
            p, clip_values=self.clip_values, device=self.device)
        if params:
            wrapper.set_params(**params)
        return wrapper

    def load_multiple(self, targets, params=None):
        """Load multiple wrappers. Returns dict {target: wrapper}."""
        return {t: self.load(t, params) for t in targets}


# ── Backward-compatible free functions ────────────────────────────────────────

def load_model_wrapper(target, models_dir, clip_values,
                       num_classes, input_dim, device, params=None):
    """Load a single-model ART wrapper for *target*."""
    return ModelLoader(models_dir, clip_values, num_classes, input_dim, device).load(target, params)

load_wrapper = load_model_wrapper


# ── CSV feature loaders ────────────────────────────────────────────────────────

def load_features_csv(path: str, label_col: str = LABEL_COL):
    """Load a feature CSV, returning (X: float32, y: int64) arrays."""
    df = pd.read_csv(path, low_memory=False)
    fc = [c for c in df.columns if c != label_col]
    X  = df[fc].values.astype(np.float32)
    y  = df[label_col].values.astype(np.int64)
    return X, y


def resolve_adv_path(component: str, attack: str, adv_dir: str,
                     dl_fallback=None) -> 'str | None':
    """Return adv CSV path for component+attack, with DL fallback if missing.

    Args:
        component:   model name (e.g. 'cat', 'lstm')
        attack:      attack name (e.g. 'pgd', 'zoo')
        adv_dir:     root adversarial directory (AT_DIR or ADV_DIR)
        dl_fallback: ordered list of DL model names to try as fallback;
                     defaults to utils.ensemble.DL_FALLBACK = ['lstm', 'resdnn']

    Returns:
        Resolved path string, or None if not found.
    """
    from .paths import adv_csv
    from .ensemble import DL_FALLBACK as _DEFAULT_FALLBACK

    if dl_fallback is None:
        dl_fallback = _DEFAULT_FALLBACK

    p = adv_csv(component, attack, adv_dir)
    if os.path.exists(p):
        return p
    for fb in dl_fallback:
        if fb == component:
            continue
        p_fb = adv_csv(fb, attack, adv_dir)
        if os.path.exists(p_fb):
            logger.debug(f"  {component}/{attack}: fallback → {fb}")
            return p_fb
    return None


# ── Ensemble / MI config parser ───────────────────────────────────────────────

def parse_ensemble_config(args) -> tuple:
    """Parse --ensemble-weights and --mi-params from argparse args.

    Returns:
        (ew, mi_cfg, w_gbt_base) — dicts and numpy array ready for use.
    """
    ew = DEFAULT_ENSEMBLE_WEIGHTS.copy()
    if getattr(args, 'ensemble_weights', None):
        ew.update(json.loads(args.ensemble_weights))

    mi_cfg = DEFAULT_MI_PARAMS.copy()
    w_gbt_base = DEFAULT_MI_W_GBT_BASE.copy()
    if getattr(args, 'mi_params', None):
        parsed = json.loads(args.mi_params)
        mi_cfg.update({k: v for k, v in parsed.items() if k != 'w_gbt_base'})
        if 'w_gbt_base' in parsed:
            w_gbt_base = np.array(parsed['w_gbt_base'], dtype=np.float64)

    return ew, mi_cfg, w_gbt_base
