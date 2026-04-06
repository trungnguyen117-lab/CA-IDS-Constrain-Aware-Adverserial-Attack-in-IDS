"""MODBUS-2023 CLI constants — targets, attacks, compatibility, AT weights."""

TREE_TARGETS = ["xgb", "cat", "rf", "et", "lgbm"]
DL_TARGETS = ["ftt"]
ALL_TARGETS = TREE_TARGETS + DL_TARGETS

WB_ATTACKS = ["fgsm", "pgd", "deepfool", "cw", "mim"]
BB_ATTACKS = ["zoo", "hsja"]
ALL_ATTACKS = WB_ATTACKS + BB_ATTACKS

# Attack compatibility for direct generation
ATTACK_COMPAT = {
    "xgb": BB_ATTACKS,
    "cat": BB_ATTACKS,
    "rf": BB_ATTACKS,
    "et": BB_ATTACKS,
    "lgbm": BB_ATTACKS,
    "ftt": WB_ATTACKS + BB_ATTACKS,
}

# Transfer sources for ML evaluation (eval only): tree models receive WB from ftt
TRANSFER_SOURCES = {
    "xgb": ["ftt_sc"],
    "cat": ["ftt_sc"],
    "rf": ["ftt_sc"],
    "et": ["ftt_sc"],
    "lgbm": ["ftt_sc"],
}

# Transfer sources for ML adversarial training
AT_TRANSFER_SOURCES = {
    "xgb": ["ftt_sc"],
    "cat": ["ftt_sc"],
    "rf": ["ftt_sc"],
    "et": ["ftt_sc"],
    "lgbm": ["ftt_sc"],
}

# Attacks that support native mask kwarg
MASK_ATTACKS = {"fgsm", "pgd", "hsja"}

# AT weighted sampling — FTT (WB + BB)
AT_WEIGHTS_FTT = {
    "fgsm": 1.5,
    "pgd": 2.5,
    "deepfool": 3.5,
    "cw": 0.5,
    "mim": 2.5,
    "hsja": 5.0,
    "zoo": 3.0,
}

# AT weighted sampling — ML (transfer WB from ftt_sc)
AT_WEIGHTS_ML = {
    "ftt_fgsm": 5.0,
    "ftt_pgd": 5.0,
    "ftt_deepfool": 5.0,
    "ftt_cw": 3.0,
    "ftt_mim": 4.5,
}

# Direct BB weights for ML AT (own zoo/hsja)
AT_BB_WEIGHT_ZOO = 7.0
AT_BB_WEIGHT_HSJA = 10.0

# Clean:Adv ratio for AT data assembly
AT_CLEAN_ADV_RATIO = {
    "xgb": 0.1,
    "cat": 0.1,
    "rf": 0.1,
    "et": 0.1,
    "lgbm": 0.1,
    "ftt": 0.5,
}

# art_generator class mapping
ATTACK_GENERATORS = {
    "fgsm": "FGSMAttackGenerator",
    "pgd": "PGDAttackGenerator",
    "deepfool": "DeepFoolAttackGenerator",
    "cw": "CWAttackGenerator",
    "jsma": "JSMAAttackGenerator",
    "zoo": "ZooAttackGenerator",
    "hsja": "HSJAAttackGenerator",
    "mim": "MIMAttackGenerator",
}

# Ensemble weights (from notebook 5 grid search)
DEFAULT_ENSEMBLE_WEIGHTS = {
    "xgb": 0.05,
    "cat": 0.0,
    "rf": 0.25,
    "et": 0.60,
    "lgbm": 0.10,
}

# MI groups
GBT_GROUP = ["xgb", "rf", "et", "lgbm"]  # cat excluded (weight=0)
DL_GROUP = ["ftt"]
