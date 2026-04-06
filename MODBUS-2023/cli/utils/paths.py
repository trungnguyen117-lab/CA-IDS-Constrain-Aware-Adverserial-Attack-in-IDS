"""Centralized path resolution for MODBUS-2023 project.

Usage:
    from utils.paths import get_path, model_path, adv_eval_dir, adv_train_dir
"""

import os

import yaml

# ── Detect MODBUS-2023 root (walk up until config.yaml found) ──
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
    """Load AT-specific attack config (config/adv_training/{attack}.yaml).
    Falls back to standard attack config if AT config doesn't exist."""
    at_path = os.path.join(ROOT, 'config', 'adv_training', f'{attack}.yaml')
    if os.path.isfile(at_path):
        with open(at_path) as f:
            return yaml.safe_load(f)
    return load_attack_config(attack)
