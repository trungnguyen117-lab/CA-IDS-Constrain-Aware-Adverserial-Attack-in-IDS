"""Config loaders for attack and training parameters.

Reads YAML files from:
  foami+/config/attacks/{attack}.yaml
  foami+/config/training/{model}.yaml
  foami+/config/adv_training/{attack}.yaml

Falls back gracefully if PyYAML is not installed or the file is missing.
"""
import os
import logging

from .paths import FOAMI_DIR

logger = logging.getLogger(__name__)


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


# ── ConfigLoader class ────────────────────────────────────────────────────────

class ConfigLoader:
    """Unified YAML config loader for attack, training, and adv_training configs.

    Usage:
        cfg = ConfigLoader()
        params = cfg.load('pgd', 'attack')
        params = cfg.load('lstm', 'training')
        params = cfg.load('pgd', 'adv_training')

        # With CLI overrides
        params = cfg.load_with_overrides('pgd', 'adv_training',
                                          overrides={'eps': 0.1, 'epochs': 50})
    """

    _CATEGORIES = {
        'attack':       os.path.join(FOAMI_DIR, 'config', 'attacks'),
        'training':     os.path.join(FOAMI_DIR, 'config', 'training'),
        'adv_training': os.path.join(FOAMI_DIR, 'config', 'adv_training'),
    }

    def load(self, name, category='attack'):
        """Load YAML config. Returns empty dict if missing."""
        if category not in self._CATEGORIES:
            raise ValueError(f"Unknown category: {category}")
        return _load_yaml(
            os.path.join(self._CATEGORIES[category], f"{name}.yaml"), name)

    def load_with_overrides(self, name, category, overrides):
        """Load config then apply CLI overrides (non-None values only)."""
        cfg = self.load(name, category)
        for k, v in overrides.items():
            if v is not None:
                cfg[k] = v
        return cfg


# ── Backward-compatible free functions ────────────────────────────────────────

_default_loader = ConfigLoader()


def load_attack_config(attack):
    """Return params dict from foami+/config/attacks/{attack}.yaml."""
    return _default_loader.load(attack, 'attack')


def load_training_config(model):
    """Return params dict from foami+/config/training/{model}.yaml."""
    return _default_loader.load(model, 'training')


def load_adv_training_config(attack):
    """Return params dict from foami+/config/adv_training/{attack}.yaml."""
    return _default_loader.load(attack, 'adv_training')
