from art.attacks.evasion import BasicIterativeMethod
from art_generator.attack_generator import SimpleAttackGenerator


class BIMAttackGenerator(SimpleAttackGenerator):
    ATTACK_CLASS = BasicIterativeMethod
    DEFAULT_PARAMS = {
        "eps": 0.2, "eps_step": 0.01, "batch_size": 64,
        "targeted": False, "max_iter": 200, "verbose": True,
    }
    MASK_MODE = "mask"
    NAME = "BIM"
