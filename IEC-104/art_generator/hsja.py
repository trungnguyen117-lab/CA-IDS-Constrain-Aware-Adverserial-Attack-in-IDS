from art.attacks.evasion import HopSkipJump
from art_generator.attack_generator import BatchedAttackGenerator


class HSJAAttackGenerator(BatchedAttackGenerator):
    ATTACK_CLASS = HopSkipJump
    DEFAULT_PARAMS = {
        "batch_size": 16, "targeted": False, "norm": 2,
        "max_iter": 10, "max_eval": 1000, "init_eval": 20,
        "init_size": 20, "verbose": True,
    }
    MASK_MODE = "mask"
    NAME = "HSJA"
