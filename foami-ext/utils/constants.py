"""Shared constants for foami+ pipeline scripts."""
import numpy as np

SINGLE_TARGETS   = ['xgb', 'cat', 'rf', 'lstm', 'resdnn']
ENSEMBLE_TARGETS = ['ensemble', 'mi']
ALL_TARGETS      = SINGLE_TARGETS + ENSEMBLE_TARGETS

GBT_TARGETS      = ['xgb', 'cat', 'rf']   # black-box attacks only
DL_TARGETS       = ['lstm', 'resdnn']      # support gradient-based attacks

ALL_ATTACKS      = ['zoo', 'deepfool', 'fgsm', 'cw', 'pgd', 'hsja', 'jsma']
BLACKBOX_ATTACKS = ['zoo', 'hsja']         # valid for GBT + ensemble/MI
WHITEBOX_ATTACKS = [a for a in ALL_ATTACKS if a not in BLACKBOX_ATTACKS]

DL_FALLBACK_TARGET = 'lstm'               # tree models use this DL model's whitebox adv CSVs

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

LOG_TAG = {
    'xgb': 'XGB', 'cat': 'CatBoost', 'rf': 'RF',
    'lstm': 'LSTM', 'resdnn': 'ResDNN',
}

_MODEL_EXT = {t: '.pth' if t in DL_TARGETS else '.pkl' for t in SINGLE_TARGETS}


def model_filename(target, suffix=''):
    """Generate checkpoint filename: framework_{target}_TVAE{suffix}.{ext}

    Examples:
        model_filename('lstm')           → 'framework_lstm_TVAE.pth'
        model_filename('cat', '_at')     → 'framework_cat_TVAE_at.pkl'
        model_filename('resdnn', '_scl') → 'framework_resdnn_TVAE_scl.pth'
    """
    return f'framework_{target}_TVAE{suffix}{_MODEL_EXT[target]}'


def model_filenames(targets=None, suffix=''):
    """Generate {target: filename} dict for multiple targets."""
    targets = targets or SINGLE_TARGETS
    return {t: model_filename(t, suffix) for t in targets}


# Backward-compatible dicts
MODEL_FILENAMES        = model_filenames()
MODEL_AT_FILENAMES     = model_filenames(suffix='_at')
MODEL_TRADES_FILENAMES = model_filenames(DL_TARGETS, suffix='_trades')
MODEL_AWP_FILENAMES    = model_filenames(DL_TARGETS, suffix='_awp')
MODEL_SCL_FILENAMES    = model_filenames(suffix='_scl')

def validate_attack_target(target, attack):
    """Raise SystemExit if attack is incompatible with target."""
    if target in GBT_TARGETS and attack not in BLACKBOX_ATTACKS:
        raise SystemExit(
            f"Target '{target}' is a tree model. "
            f"Only black-box attacks are supported: {BLACKBOX_ATTACKS}. "
            f"Got: {attack}"
        )
    if target in ENSEMBLE_TARGETS and attack not in BLACKBOX_ATTACKS:
        raise SystemExit(
            f"Ensemble/MI targets require black-box attacks: {BLACKBOX_ATTACKS}. "
            f"Got: {attack}"
        )
