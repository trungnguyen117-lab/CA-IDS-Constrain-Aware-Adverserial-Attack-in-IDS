"""IEC-104 CLI constants — targets, attacks, compatibility, AT weights."""

TREE_TARGETS = ["cat", "rf"]
DL_TARGETS = ["lstm", "resdnn"]
ALL_TARGETS = TREE_TARGETS + DL_TARGETS

WB_ATTACKS = ["fgsm", "pgd", "deepfool", "cw", "mim", "jsma"]
BB_ATTACKS = ["zoo", "hsja"]
ALL_ATTACKS = WB_ATTACKS + BB_ATTACKS

# Attack compatibility for direct generation
ATTACK_COMPAT = {
    "cat": BB_ATTACKS,
    "rf": BB_ATTACKS,
    "lstm": WB_ATTACKS + BB_ATTACKS,
    "resdnn": WB_ATTACKS + BB_ATTACKS,
}

# Transfer sources for ML evaluation (eval only)
TRANSFER_SOURCES = {
    "cat": ["resdnn_sc"],
    "rf": ["resdnn_sc"],
}

# Transfer sources for ML adversarial training (superset of TRANSFER_SOURCES)
AT_TRANSFER_SOURCES = {
    "cat": ["resdnn_sc", "lstm_sc"],
    "rf": ["resdnn_sc", "lstm_sc"],
}

# Attacks that support native mask kwarg
MASK_ATTACKS = {"fgsm", "pgd", "hsja"}

# AT weighted sampling — LSTM (WB + BB, tuned by ASR)
AT_WEIGHTS_LSTM = {
    "fgsm": 1.5,
    "pgd": 2.5,
    "deepfool": 3.5,
    "cw": 0.5,
    "mim": 2.5,
    "hsja": 5.0,
    "zoo": 3.0,
}

# AT weighted sampling — ResDNN (WB + BB, tuned by ASR)
AT_WEIGHTS_RESDNN = {
    "fgsm": 1.5,
    "pgd": 2.5,
    "deepfool": 4.0,
    "cw": 0.5,
    "zoo": 1.0,
    "hsja": 1.0,
    "mim": 0.5
}

# AT weighted sampling — ML (transfer WB from resdnn_sc only, no BB transfer)
# Weights tuned by ASR: hsja(~32%) > deepfool(~20%) > fgsm(~18%) > pgd/mim(~15%) > cw(~10%)
AT_WEIGHTS_ML = {
    "resdnn_fgsm": 5.0,
    "resdnn_pgd": 5.0,
    "resdnn_deepfool": 5.0,
    "resdnn_cw": 5.0,
    "resdnn_mim": 5.0,
    "resdnn_jsma": 5.0,
    "lstm_hsja": 3.0,
    "lstm_fgsm": 5.0
}

# Direct BB weights for ML AT (cat/rf own zoo/hsja)
AT_BB_WEIGHT_ZOO = 7.0
AT_BB_WEIGHT_HSJA = 5.0

# Clean:Adv ratio for AT data assembly
AT_CLEAN_ADV_RATIO = {
    "cat": 0.05,      # 1:10 clean:adv (matches notebook)
    "rf": 0.05,       # 1:10 clean:adv (matches notebook)
    "lstm": 0.2,     # 1:5 clean:adv (more adv budget for BB attacks)
    "resdnn": 0.1 ,   # 1:1 clean:adv
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

