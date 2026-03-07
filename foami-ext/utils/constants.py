"""Shared constants for foami+ pipeline scripts."""
import numpy as np

SINGLE_TARGETS   = ['xgb', 'cat', 'rf', 'lstm', 'resdnn']
ENSEMBLE_TARGETS = ['ensemble', 'mi']
ALL_TARGETS      = SINGLE_TARGETS + ENSEMBLE_TARGETS

GBT_TARGETS      = ['xgb', 'cat', 'rf']   # black-box attacks only
DL_TARGETS       = ['lstm', 'resdnn']      # support gradient-based attacks

ALL_ATTACKS      = ['zoo', 'deepfool', 'fgsm', 'cw', 'pgd', 'hsja', 'jsma']
BLACKBOX_ATTACKS = ['zoo', 'hsja']         # valid for GBT + ensemble/MI

# Ensemble weights from notebook 8 grid search
DEFAULT_ENSEMBLE_WEIGHTS = {
    'xgb':    0.0,
    'cat':    0.25,
    'rf':     0.35,
    'lstm':   0.1,
    'resdnn': 0.3,
}

# MI base weights (cat, rf) from notebook 9
DEFAULT_MI_W_GBT_BASE = np.array([0.42, 0.58])
DEFAULT_MI_PARAMS     = {'alpha': 0.0, 'beta': 0.0, 'threshold': 0.5}

LABEL_COL = 'Label'

MODEL_FILENAMES = {
    'xgb':    'framework_xgb_TVAE.pkl',
    'cat':    'framework_cat_TVAE.pkl',
    'rf':     'framework_rf_TVAE.pkl',
    'lstm':   'framework_lstm_TVAE.pth',
    'resdnn': 'framework_resdnn_TVAE.pth',
}

MODEL_AT_FILENAMES = {
    'xgb':    'framework_xgb_TVAE_at.pkl',
    'cat':    'framework_cat_TVAE_at.pkl',
    'rf':     'framework_rf_TVAE_at.pkl',
    'lstm':   'framework_lstm_TVAE_at.pth',
    'resdnn': 'framework_resdnn_TVAE_at.pth',
}
