"""Train surrogate ResDNN → generate adversarial on TEST set for eval.

Supports both WB attacks (fgsm, pgd, deepfool, cw, mim) and BB attacks
(zoo, hsja) on the surrogate model. BB attacks on surrogate test whether
query-based attacks transfer to target models.

Usage:
    python 5_generate_surrogate_adv.py                          # all WB attacks
    python 5_generate_surrogate_adv.py --attacks fgsm pgd       # specific attacks
    python 5_generate_surrogate_adv.py --attacks zoo hsja       # BB on surrogate
    python 5_generate_surrogate_adv.py --device cuda            # GPU
    python 5_generate_surrogate_adv.py --skip-train             # reuse saved surrogate

Output:
    adv_samples/adv_eval/surrogate_resdnn/surrogate_resdnn_{attack}_adv.csv
"""

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

# Path bootstrap
_HERE = os.path.dirname(os.path.abspath(__file__))
_CLI = os.path.dirname(_HERE)
_IEC = os.path.dirname(_CLI)
sys.path.insert(0, _IEC)
sys.path.insert(0, os.path.join(_IEC, "script"))
sys.path.insert(0, _CLI)

from utils.logging import setup_logging, get_logger
from utils.constants import WB_ATTACKS, BB_ATTACKS, ALL_ATTACKS, ATTACK_GENERATORS
from utils.paths import load_attack_config, adv_eval_dir, model_path
from utils.loaders import load_dataset
from utils.masking import get_mutate_indices
from model.surrogate_resdnn import SurrogateResDNNModel

logger = get_logger(__name__)

SURROGATE_NAME = "surrogate_resdnn"
SURROGATE_FILE = f"framework_{SURROGATE_NAME}.pth"


def _get_generator(attack_name, classifier, config):
    import art_generator as ag
    cls_name = ATTACK_GENERATORS[attack_name]
    cls = getattr(ag, cls_name)
    return cls(classifier, generator_params=config)


def train_surrogate(X_train, y_train, device, save_path):
    """Train surrogate and save checkpoint."""
    logger.info("Training surrogate ResDNN ...")
    m = SurrogateResDNNModel()
    m.train(X_train, y_train, device=device)
    m.save(save_path)
    logger.info(f"Surrogate saved → {save_path}")
    return m


def generate_adv(surrogate, attacks, X_test, y_test, feature_names, mutate_indices, device):
    """Generate adversarial in raw space (InputNorm embedded in model)."""
    # ART classifier — InputNorm handles normalization internally
    art_clf = surrogate.wrap_for_art(X_test, device=device)
    metadata = {"feature_names": feature_names, "label_column": "Label"}

    # Report surrogate accuracy
    preds_clean = np.argmax(art_clf.predict(X_test), axis=1)
    from sklearn.metrics import accuracy_score, f1_score
    acc = accuracy_score(y_test, preds_clean)
    f1 = f1_score(y_test, preds_clean, average="macro")
    logger.info(f"Surrogate clean acc={acc*100:.2f}%, F1={f1:.4f}")

    out_dir = adv_eval_dir(SURROGATE_NAME)
    os.makedirs(out_dir, exist_ok=True)

    for atk_name in attacks:
        logger.info(f"{'='*50}")
        logger.info(f"Generating {atk_name.upper()} from surrogate ResDNN")
        logger.info(f"{'='*50}")

        atk_cfg = load_attack_config(atk_name)
        gen = _get_generator(atk_name, art_clf, atk_cfg)

        start = time.time()
        df_adv = gen.generate(X_test, y_test, metadata, mutate_indices)
        elapsed = time.time() - start
        logger.info(f"{atk_name.upper()} runtime: {elapsed:.2f}s")

        X_adv = df_adv[feature_names].values.astype(np.float32)

        # Save
        filename = f"{SURROGATE_NAME}_{atk_name}_adv.csv"
        out_path = os.path.join(out_dir, filename)
        df_adv.to_csv(out_path, index=False)
        logger.info(f"Saved -> {out_path}")

        # Quick ASR on surrogate itself
        preds_adv = np.argmax(art_clf.predict(X_adv), axis=1)
        correct = np.where(y_test == preds_clean)[0]
        if len(correct) > 0:
            asr = np.sum(preds_clean[correct] != preds_adv[correct]) / len(correct) * 100
            acc_adv = accuracy_score(y_test, preds_adv) * 100
            logger.info(f"{atk_name.upper()} -> surrogate ASR={asr:.2f}%, adv acc={acc_adv:.2f}%")


def main():
    parser = argparse.ArgumentParser(description="Generate adv from surrogate ResDNN")
    parser.add_argument("--attacks", "-a", nargs="+", default=WB_ATTACKS,
                        choices=ALL_ATTACKS + ["all"],
                        help="Attacks to generate (default: WB only). "
                             "BB attacks (zoo, hsja) also supported on surrogate.")
    parser.add_argument("--device", "-d", default="cpu",
                        choices=["cpu", "cuda", "mps", "auto"])
    parser.add_argument("--skip-train", action="store_true",
                        help="Load existing surrogate instead of retraining")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    setup_logging(args.log_level)

    if args.device == "auto":
        import torch
        if torch.cuda.is_available():
            args.device = "cuda"
        elif torch.backends.mps.is_available():
            args.device = "mps"
        else:
            args.device = "cpu"

    attacks = ALL_ATTACKS if "all" in args.attacks else args.attacks

    # Load datasets
    _, X_train, y_train, _ = load_dataset("train_tvae")
    df_test, X_test, y_test, feature_names = load_dataset("test")
    # Always compute mutate_indices from TRAIN for consistency
    df_train_ref, *_ = load_dataset("train")
    mutate_indices = get_mutate_indices(df_train_ref)
    logger.info(f"Train: {X_train.shape}, Test: {X_test.shape}")
    logger.info(f"Protected features: {len(mutate_indices)} indices")

    # Train or load surrogate
    save_path = model_path(SURROGATE_FILE)
    if args.skip_train and os.path.exists(save_path):
        logger.info(f"Loading existing surrogate from {save_path}")
        surrogate = SurrogateResDNNModel.load(save_path, device=args.device)
    else:
        surrogate = train_surrogate(X_train, y_train, args.device, save_path)

    # Generate adversarial
    generate_adv(surrogate, attacks, X_test, y_test, feature_names,
                 mutate_indices, args.device)

    logger.info("Done!")


if __name__ == "__main__":
    main()
