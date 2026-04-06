import numpy as np
from art.attacks.evasion import MomentumIterativeMethod
from art_generator.attack_generator import SimpleAttackGenerator


class MIMAttackGenerator(SimpleAttackGenerator):
    ATTACK_CLASS = MomentumIterativeMethod
    DEFAULT_PARAMS = {
        "norm": np.inf, "eps": 0.3, "eps_step": 0.01, "decay": 1.0,
        "max_iter": 200, "targeted": False, "batch_size": 64, "verbose": True,
    }
    MASK_MODE = "mask"
    NAME = "MIM"
