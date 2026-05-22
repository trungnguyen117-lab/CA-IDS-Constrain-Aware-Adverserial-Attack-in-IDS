"""JSMA — Jacobian-based Saliency Map Attack."""

from __future__ import annotations

from art.attacks.evasion import SaliencyMapMethod

from .base import BatchedAttackGenerator


class JSMAAttackGenerator(BatchedAttackGenerator):
    ATTACK_CLASS = SaliencyMapMethod
    DEFAULT_PARAMS = {"theta": 0.02, "gamma": 0.05, "batch_size": 64, "verbose": True}
    MASK_MODE = "post_apply"
    NAME = "JSMA"
