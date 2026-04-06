from art.attacks.evasion import SaliencyMapMethod
from art_generator.attack_generator import BatchedAttackGenerator


class JSMAAttackGenerator(BatchedAttackGenerator):
    ATTACK_CLASS = SaliencyMapMethod
    DEFAULT_PARAMS = {
        "theta": 0.02, "gamma": 0.05, "batch_size": 64, "verbose": True,
    }
    MASK_MODE = "post_apply"
    NAME = "JSMA"
