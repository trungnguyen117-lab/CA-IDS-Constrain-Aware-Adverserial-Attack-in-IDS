from art.attacks.evasion import ProjectedGradientDescentPyTorch
from art_generator.attack_generator import SimpleAttackGenerator


class PGDAttackGenerator(SimpleAttackGenerator):
    ATTACK_CLASS = ProjectedGradientDescentPyTorch
    DEFAULT_PARAMS = {
        "eps": 0.2, "eps_step": 0.01, "batch_size": 32,
        "targeted": False, "max_iter": 200, "verbose": True,
    }
    MASK_MODE = "mask"
    NAME = "PGD"
