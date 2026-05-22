"""HSJA — HopSkipJump decision-based attack."""

from __future__ import annotations

from art.attacks.evasion import HopSkipJump

from .base import BatchedAttackGenerator


class HSJAAttackGenerator(BatchedAttackGenerator):
    ATTACK_CLASS = HopSkipJump
    DEFAULT_PARAMS = {
        "batch_size": 16, "targeted": False, "norm": 2,
        "max_iter": 10, "max_eval": 1000, "init_eval": 20,
        "init_size": 20, "verbose": True,
    }
    NAME = "HSJA"
