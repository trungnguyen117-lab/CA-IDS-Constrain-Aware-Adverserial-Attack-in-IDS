"""DeepFool — minimum-perturbation attack."""

from __future__ import annotations

from art.attacks.evasion import DeepFool

from .base import SimpleAttackGenerator


class DeepFoolAttackGenerator(SimpleAttackGenerator):
    ATTACK_CLASS = DeepFool
    DEFAULT_PARAMS = {
        "max_iter": 100, "batch_size": 64, "nb_grads": 5, "epsilon": 1e-6,
    }
    MASK_MODE = "post_apply"
    NAME = "DeepFool"

    def attack_init_kwargs(self, attack_params):
        return {"classifier": self.classifier, **attack_params}
