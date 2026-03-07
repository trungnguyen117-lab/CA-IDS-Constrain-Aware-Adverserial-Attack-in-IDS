"""Config loaders for attack and training parameters.

Reads YAML files from:
  foami+/config/attacks/{attack}.yaml
  foami+/config/training/{model}.yaml

Falls back gracefully if PyYAML is not installed or the file is missing.
"""
import os
import logging

from .paths import FOAMI_DIR

logger = logging.getLogger(__name__)

_CONFIG_DIR          = os.path.join(FOAMI_DIR, 'config', 'attacks')
_TRAINING_CONFIG_DIR = os.path.join(FOAMI_DIR, 'config', 'training')
_AT_CONFIG_DIR       = os.path.join(FOAMI_DIR, 'config', 'adv_training')


def _load_yaml(path: str, label: str) -> dict:
    if not os.path.exists(path):
        logger.debug(f"[config] No config file for '{label}' at {path}")
        return {}
    try:
        import yaml
    except ImportError:
        logger.warning(
            "[config] PyYAML not installed — config files are ignored. "
            "Install with: pip install pyyaml"
        )
        return {}
    try:
        with open(path, 'r') as fh:
            cfg = yaml.safe_load(fh) or {}
        logger.debug(f"[config] Loaded '{label}' config from {path}: {cfg}")
        return cfg
    except Exception as exc:
        logger.warning(f"[config] Failed to parse {path} ({label}): {exc}")
        return {}


def load_attack_config(attack: str) -> dict:
    """Return params dict from foami+/config/attacks/{attack}.yaml.

    Priority when used together with make_generator:
      1. CLI --attack-params  (highest)
      2. config YAML file     (this function)
      3. hardcoded defaults inside each ART generator class  (lowest)

    Returns an empty dict if the file is missing or PyYAML is unavailable
    (caller then falls back to the hardcoded defaults).
    """
    return _load_yaml(os.path.join(_CONFIG_DIR, f"{attack}.yaml"), attack)


def load_training_config(model: str) -> dict:
    """Return params dict from foami+/config/training/{model}.yaml.

    Used by train_tree.py and train_dl.py. The returned dict is passed
    directly as constructor kwargs to the model class (e.g. XGBModel,
    LSTMModel). Runtime-only values (device, num_class, input_dim) are
    NOT stored in config — those are injected by the training script.

    Returns an empty dict if the file is missing or PyYAML is unavailable
    (caller then falls back to the hardcoded defaults in the training script).
    """
    return _load_yaml(os.path.join(_TRAINING_CONFIG_DIR, f"{model}.yaml"), model)


def load_adv_training_config(attack: str) -> dict:
    """Return params dict from foami+/config/adv_training/{attack}.yaml.

    Used by adv_train_dl.py. CLI flags have higher priority and override values
    returned here (e.g. --eps, --max-iter).

    Returns an empty dict if the file is missing or PyYAML is unavailable.
    """
    return _load_yaml(os.path.join(_AT_CONFIG_DIR, f"{attack}.yaml"), attack)
