from art.attacks.evasion import CarliniL2Method
from art_generator.attack_generator import SimpleAttackGenerator


class CWAttackGenerator(SimpleAttackGenerator):
    ATTACK_CLASS = CarliniL2Method
    DEFAULT_PARAMS = {
        "confidence": 0.0, "learning_rate": 0.01, "binary_search_steps": 3,
        "max_iter": 3, "batch_size": 64, "verbose": False,
        "initial_const": 0.01, "max_halving": 5, "max_doubling": 5,
    }
    MASK_MODE = "post_apply"
    NAME = "CW"

    def _attack_init_kwargs(self, attack_params):
        return {"classifier": self.classifier, **attack_params}
