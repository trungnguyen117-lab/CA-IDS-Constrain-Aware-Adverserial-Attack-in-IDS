"""
Centralized path resolution for IEC-104 project.

Usage:
    from utils.paths import get_path, model_path, adv_eval_dir, adv_train_dir

    train_csv  = get_path('train')           # .../IEC-104/datasets/train_shap_66.csv
    models_dir = get_path('models')          # .../IEC-104/training/models
    pth_file   = model_path('framework_resdnn_TVAE.pth')
    adv_csv    = adv_eval_dir('resdnn', 'resdnn_pgd_adv.csv')
"""

import os

import yaml

# ── Detect IEC-104 root (walk up until config.yaml found) ──
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))


def _find_root(start):
    d = start
    for _ in range(10):
        if os.path.isfile(os.path.join(d, 'config.yaml')):
            return d
        parent = os.path.dirname(d)
        if parent == d:
            break
        d = parent
    raise FileNotFoundError(f'config.yaml not found walking up from {start}')


ROOT = _find_root(_THIS_DIR)

with open(os.path.join(ROOT, 'config.yaml')) as f:
    cfg = yaml.safe_load(f)

_paths = cfg.get('paths', {})
_version_dataset = cfg.get('version_dataset', {})

# ── Version support ──
_version = None


def set_version(version):
    """Set active version tag. Called once at CLI startup.

    v1 (default) → existing paths unchanged.
    v2+ → versioned dataset, models, adv_samples, AT configs.
    Version-dataset mapping defined in config.yaml `version_dataset`.
    """
    global _version
    _version = version
    if version and version != "v1":
        # Switch training dataset via version_dataset mapping
        ds_key = _version_dataset.get(version)
        if ds_key and ds_key in _paths:
            _paths['train_tvae'] = _paths[ds_key]

        # Versioned directories
        _paths['models'] = f"training/models_{version}"
        _paths['adv_eval'] = f"adv_samples/{version}/adv_eval"
        _paths['adv_training'] = f"adv_samples/{version}/adv_training"
        _paths['models_at'] = f"defense/models_{version}"


def get_version():
    """Return current version tag."""
    return _version or "v1"


def adv_training_config_dir():
    """AT config directory — version-aware."""
    v = get_version()
    if v == "v1":
        return os.path.join(ROOT, 'config', 'adv_training')
    return os.path.join(ROOT, 'config', f'adv_training_{v}')


def model_stem(model_name, at=False):
    """Return model filename stem (no extension) based on active version.

    v1:  framework_{model}_TVAE / framework_{model}_TVAE_at
    v2+: framework_{model}_TVAE_{version} / framework_{model}_TVAE_{version}_at
    """
    v = get_version()
    tag = f"_{v}" if v != "v1" else ""
    suffix = "_at" if at else ""
    return f"framework_{model_name}_TVAE{tag}{suffix}"


def get_path(key):
    """Resolve a config path key to absolute path."""
    if key not in _paths:
        raise KeyError(f'Unknown path key: {key!r}. Available: {list(_paths.keys())}')
    return os.path.join(ROOT, _paths[key])


def model_path(filename, at=False):
    """Path to model file. at=True → defense/models/"""
    base = get_path('models_at') if at else get_path('models')
    return os.path.join(base, filename)


def adv_eval_dir(target, filename=''):
    """adv_samples/adv_eval/{target}/{filename}"""
    return os.path.join(get_path('adv_eval'), target, filename)


def adv_train_dir(target, filename=''):
    """adv_samples/adv_training/{target}/{filename}"""
    return os.path.join(get_path('adv_training'), target, filename)


def attack_config_path(attack):
    """config/attacks/{attack}.yaml"""
    return os.path.join(get_path('attack_config'), f'{attack}.yaml')


def training_config_path(model_name):
    """config/training/{model_name}.yaml"""
    return os.path.join(get_path('training_config'), f'{model_name}.yaml')


def load_attack_config(attack):
    """Load attack YAML config and return as dict."""
    path = attack_config_path(attack)
    with open(path) as f:
        return yaml.safe_load(f)


def load_adv_training_config(attack):
    """Load AT-specific attack config (version-aware).

    Search order: config/adv_training_{version}/ → config/adv_training/ → config/attacks/
    """
    at_path = os.path.join(adv_training_config_dir(), f'{attack}.yaml')
    if os.path.isfile(at_path):
        with open(at_path) as f:
            return yaml.safe_load(f)
    return load_attack_config(attack)
