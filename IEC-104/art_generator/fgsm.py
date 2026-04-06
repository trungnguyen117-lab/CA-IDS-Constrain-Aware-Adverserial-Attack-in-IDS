from art.attacks.evasion import FastGradientMethod
from art_generator.attack_generator import SimpleAttackGenerator


class FGSMAttackGenerator(SimpleAttackGenerator):
    ATTACK_CLASS = FastGradientMethod
    DEFAULT_PARAMS = {"eps": 0.1, "batch_size": 64, "eps_step": 0.1, "targeted": False}
    MASK_MODE = "mask"
    NAME = "FGSM"
