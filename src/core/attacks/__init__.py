"""Adversarial attacks: per-attack generators + ART wrappers.

Public API:
  - ATTACK_REGISTRY: name → AttackGenerator class
  - make_attack(name, classifier, params)
  - wrap_for_art(cfg, model, target, X_ref, device, preprocessing_defences)
  - AttackGenerator / SimpleAttackGenerator / BatchedAttackGenerator (base classes)
"""

from __future__ import annotations

from .base import AttackGenerator, BatchedAttackGenerator, SimpleAttackGenerator
from .cw import CWAttackGenerator
from .deepfool import DeepFoolAttackGenerator
from .fgsm import FGSMAttackGenerator
from .hsja import HSJAAttackGenerator
from .jsma import JSMAAttackGenerator
from .pgd import PGDAttackGenerator
from .wrap import get_clip_values, wrap_for_art
from .zoo import ZooAttackGenerator

ATTACK_REGISTRY: dict[str, type[AttackGenerator]] = {
    "fgsm":     FGSMAttackGenerator,
    "pgd":      PGDAttackGenerator,
    "cw":       CWAttackGenerator,
    "deepfool": DeepFoolAttackGenerator,
    "jsma":     JSMAAttackGenerator,
    "hsja":     HSJAAttackGenerator,
    "zoo":      ZooAttackGenerator,
}


def make_attack(name: str, classifier, params: dict | None = None) -> AttackGenerator:
    if name not in ATTACK_REGISTRY:
        raise ValueError(f"Unknown attack {name!r}. Choose from: {sorted(ATTACK_REGISTRY)}")
    return ATTACK_REGISTRY[name](classifier, generator_params=params)


__all__ = [
    "AttackGenerator", "SimpleAttackGenerator", "BatchedAttackGenerator",
    "FGSMAttackGenerator", "PGDAttackGenerator", "CWAttackGenerator",
    "DeepFoolAttackGenerator", "JSMAAttackGenerator", "HSJAAttackGenerator",
    "ZooAttackGenerator",
    "ATTACK_REGISTRY", "make_attack",
    "wrap_for_art", "get_clip_values"
]
