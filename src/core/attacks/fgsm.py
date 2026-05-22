"""FGSM — Fast Gradient Sign Method."""

from __future__ import annotations

from art.attacks.evasion import FastGradientMethod

from .base import SimpleAttackGenerator


class FGSMAttackGenerator(SimpleAttackGenerator):
    ATTACK_CLASS = FastGradientMethod
    DEFAULT_PARAMS = {"eps": 0.1, "batch_size": 64, "eps_step": 0.1, "targeted": False}
    NAME = "FGSM"
