"""Zoo — zeroth-order optimization black-box attack."""

from __future__ import annotations

from art.attacks.evasion import ZooAttack

from .base import SimpleAttackGenerator


class ZooAttackGenerator(SimpleAttackGenerator):
    ATTACK_CLASS = ZooAttack
    DEFAULT_PARAMS = {
        "confidence": 0.0, "targeted": False, "learning_rate": 1e-1,
        "max_iter": 100, "binary_search_steps": 3, "initial_const": 1e-3,
        "abort_early": True, "use_resize": False, "use_importance": False,
        "nb_parallel": 10, "batch_size": 1, "variable_h": 0.02, "verbose": True,
    }
    MASK_MODE = "post_apply"
    NAME = "Zoo"

    def attack_init_kwargs(self, attack_params):
        return {"classifier": self.classifier, **attack_params}

    def generate_raw(self, x, mask):
        return self.attack.generate(x)
