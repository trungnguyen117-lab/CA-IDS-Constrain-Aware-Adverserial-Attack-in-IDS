"""Model loading helpers shared by generate and evaluate pipeline scripts.

All wrappers return ART-compatible estimator objects ready for adversarial
generation or evaluation.
"""
import os
import logging

import joblib
import numpy as np
import pandas as pd

from .constants import (
    SINGLE_TARGETS,
    DEFAULT_ENSEMBLE_WEIGHTS, DEFAULT_MI_W_GBT_BASE, DEFAULT_MI_PARAMS,
)

logger = logging.getLogger(__name__)


# ── File guard ────────────────────────────────────────────────────────────────

def require_file(path: str) -> None:
    """Raise SystemExit if *path* does not exist."""
    if not os.path.exists(path):
        raise SystemExit(f"File not found: {path}")


# ── Single-model wrapper loader ───────────────────────────────────────────────

def _load_sklearn_wrapper(path: str, clip_values, num_classes: int, input_dim: int):
    from art_classifier.sklearn_classifier import SkleanWrapper
    model = joblib.load(path)
    return SkleanWrapper(
        model=model,
        num_classes=num_classes,
        input_shape=(input_dim,),
        clip_values=clip_values,
    )


def load_wrapper(target: str, models_dir: str, clip_values,
                 num_classes: int, input_dim: int, device: str):
    """Load a single-model ART wrapper for *target*."""
    if target == 'xgb':
        from art_classifier.xgb_classifier import XGBWrapper
        p = os.path.join(models_dir, 'framework_xgb_TVAE.pkl')
        require_file(p)
        try:
            model = joblib.load(p)
            return XGBWrapper(model=model, num_classes=num_classes,
                              input_shape=(input_dim,), clip_values=clip_values)
        except Exception as e:
            raise SystemExit(
                f"XGB load failed: {e}. Try: uv pip install 'xgboost<2.0'"
            ) from e

    if target == 'cat':
        from art_classifier.catb_classifier import CatBoostWrapper
        p = os.path.join(models_dir, 'framework_cat_TVAE.pkl')
        require_file(p)
        try:
            model = joblib.load(p)
            return CatBoostWrapper(model=model, num_classes=num_classes,
                                   input_shape=(input_dim,), clip_values=clip_values,
                                   device=device)
        except Exception:
            return _load_sklearn_wrapper(p, clip_values, num_classes, input_dim)

    if target == 'rf':
        p = os.path.join(models_dir, 'framework_rf_TVAE.pkl')
        require_file(p)
        return _load_sklearn_wrapper(p, clip_values, num_classes, input_dim)

    if target == 'lstm':
        from art_classifier.lstm_classifier import LSTMWrapper
        p = os.path.join(models_dir, 'framework_lstm_TVAE.pth')
        require_file(p)
        return LSTMWrapper.from_checkpoint(p, clip_values=clip_values, device=device)

    if target == 'resdnn':
        from art_classifier.resdnn_classifier import ResDNNWrapper
        p = os.path.join(models_dir, 'framework_resdnn_TVAE.pth')
        require_file(p)
        return ResDNNWrapper.from_checkpoint(p, clip_values=clip_values, device=device)

    raise ValueError(f"Unknown target: {target}")


# ── CSV feature loaders ────────────────────────────────────────────────────────

def load_features_csv(path: str, label_col: str = 'Label'):
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


# ── Ensemble / MI predictor builder (used by evaluate) ───────────────────────

class WrapperAdapter:
    """Thin adapter so a wrapper's predict_proba looks like predict."""
    def __init__(self, wrapper):
        self._w = wrapper

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._w.predict_proba(X)


def build_predictor(target: str, models_dir: str, clip_values,
                    num_classes: int, input_dim: int, device: str,
                    ew: dict = None, mi_cfg: dict = None,
                    w_gbt_base: np.ndarray = None):
    """Return a predictor (has .predict(X) → proba) for *target*."""
    from art_classifier.ensemble_classifier import EnsembleEstimator
    from art_classifier.mi_classifier import MIEstimator

    ew         = ew         or DEFAULT_ENSEMBLE_WEIGHTS.copy()
    mi_cfg     = mi_cfg     or DEFAULT_MI_PARAMS.copy()
    w_gbt_base = w_gbt_base if w_gbt_base is not None else DEFAULT_MI_W_GBT_BASE.copy()

    if target in SINGLE_TARGETS:
        return WrapperAdapter(load_wrapper(target, models_dir, clip_values,
                                          num_classes, input_dim, device))

    if target == 'ensemble':
        wrappers = {}
        for t in SINGLE_TARGETS:
            if ew.get(t, 0.0) > 0:
                logger.info(f"  Loading {t} ...")
                wrappers[t] = load_wrapper(t, models_dir, clip_values,
                                           num_classes, input_dim, device)
        return EnsembleEstimator(wrappers=wrappers, weights=ew,
                                 num_classes=num_classes, clip_values=clip_values)

    if target == 'mi':
        logger.info("  Loading GBT (cat, rf) ...")
        gbt = {
            'cat': load_wrapper('cat', models_dir, clip_values, num_classes, input_dim, device),
            'rf':  load_wrapper('rf',  models_dir, clip_values, num_classes, input_dim, device),
        }
        logger.info("  Loading DL (lstm, resdnn) ...")
        dl = {
            'lstm':   load_wrapper('lstm',   models_dir, clip_values, num_classes, input_dim, device),
            'resdnn': load_wrapper('resdnn', models_dir, clip_values, num_classes, input_dim, device),
        }
        return MIEstimator(gbt_wrappers=gbt, dl_wrappers=dl,
                           num_classes=num_classes, clip_values=clip_values,
                           w_gbt_base=w_gbt_base, **mi_cfg)

    raise ValueError(f"Unknown target: {target}")
