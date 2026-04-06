"""Evaluate surrogate ResDNN adversarial transferability to all target models.

Usage:
    python 6_evaluate_surrogate.py                            # all targets, baseline
    python 6_evaluate_surrogate.py --at both                  # compare baseline vs AT
    python 6_evaluate_surrogate.py --target lstm resdnn       # specific targets
    python 6_evaluate_surrogate.py --target all --at both     # full comparison

Output: Rich table showing clean acc + ASR per attack per target model.
"""

import argparse
import glob
import os
import sys

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
from utils.constants import ALL_TARGETS, WB_ATTACKS
from utils.evaluation import report_metrics, compute_asr, print_summary, print_comparison_all
from utils.loaders import load_dataset
from utils.paths import model_path, adv_eval_dir, model_stem
from utils.defense import create_preprocessing_defense
from model import MODEL_REGISTRY

logger = get_logger(__name__)

SURROGATE_NAME = "surrogate_resdnn"


def _load_model(target, at=False, device="cpu"):
    ext = ".pkl" if target in ("cat", "rf") else ".pth"
    path = model_path(f"{model_stem(target, at=at)}{ext}", at=at)
    if not os.path.exists(path):
        logger.warning(f"Model not found: {path}")
        return None
    return MODEL_REGISTRY[target].load(path, device=device)


def _find_surrogate_csvs():
    """Find all surrogate_resdnn adversarial CSVs."""
    d = adv_eval_dir(SURROGATE_NAME)
    results = {}
    if not os.path.isdir(d):
        logger.warning(f"Surrogate adv dir not found: {d}")
        return results
    for f in sorted(glob.glob(os.path.join(d, f"{SURROGATE_NAME}_*_adv.csv"))):
        basename = os.path.basename(f).replace("_adv.csv", "")
        prefix = f"{SURROGATE_NAME}_"
        atk_name = basename[len(prefix):]
        results[atk_name] = f
    return results


def evaluate_target(target, adv_files, X_test, y_test, feature_names, at=False,
                    device="cpu", defense=None, sigma=0.01):
    """Evaluate a target model on surrogate adversarial samples."""
    tag = f"{target.upper()} {'(AT)' if at else '(baseline)'}"
    if defense:
        tag += f" +{defense}(σ={sigma})"
    logger.info(f"{'='*60}")
    logger.info(f"Evaluating: {tag} vs Surrogate ResDNN")
    logger.info(f"{'='*60}")

    m = _load_model(target, at=at, device=device)
    if m is None:
        return {}

    # Always use ART classifier for evaluation; defense is applied when specified
    preprocessing_defences = create_preprocessing_defense(defense, sigma=sigma)
    art_clf = m.wrap_for_art(
        X_test, preprocessing_defences=preprocessing_defences, device=device,
    )
    predict_fn = lambda X: np.argmax(art_clf.predict(X), axis=1)
    if defense:
        logger.info(f"Defense active: {defense} (sigma={sigma})")

    y_clean = predict_fn(X_test)
    clean_metrics = report_metrics(f"{tag} clean", y_test, y_clean)
    results = {"clean": clean_metrics}

    for atk_name, path in adv_files.items():
        try:
            df_adv = pd.read_csv(path)
            X_adv = df_adv[feature_names].values.astype(np.float32)
            y_adv_true = df_adv["Label"].values.astype(int)

            y_adv_pred = predict_fn(X_adv)
            metrics = report_metrics(f"{tag} surr_{atk_name}", y_adv_true, y_adv_pred)

            asr = compute_asr(y_test, y_clean, y_adv_pred)
            metrics["asr"] = asr
            logger.info(f"{'':>25s}  ASR={asr:6.2f}%")
            results[f"surr_{atk_name}"] = metrics
        except Exception as e:
            logger.error(f"Failed {atk_name}: {e}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate surrogate ResDNN transferability",
    )
    parser.add_argument("--target", "-t", nargs="+", default=["all"],
                        choices=ALL_TARGETS + ["all"],
                        help="Target models to evaluate (default: all)")
    parser.add_argument("--at", default="false",
                        choices=["false", "true", "both"],
                        help="Evaluate: false=baseline, true=AT, both=compare")
    parser.add_argument("--device", "-d", default="cpu",
                        choices=["cpu", "cuda", "auto"])
    parser.add_argument("--defense", default=None,
                        choices=["gaussian_noise", "feature_squeezing"],
                        help="Preprocessing defense to apply during inference")
    parser.add_argument("--sigma", type=float, default=0.01,
                        help="Gaussian noise sigma (default: 0.01)")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    setup_logging(args.log_level)

    if args.device == "auto":
        import torch
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    targets = ALL_TARGETS if "all" in args.target else args.target

    # Load test data
    _, X_test, y_test, feature_names = load_dataset("test")

    # Find surrogate adv CSVs
    adv_files = _find_surrogate_csvs()
    if not adv_files:
        logger.error("No surrogate adversarial CSVs found. Run 5_generate_surrogate_adv.py first.")
        return

    logger.info(f"Found {len(adv_files)} surrogate attacks: {list(adv_files.keys())}")

    defense_kwargs = dict(defense=args.defense, sigma=args.sigma)

    if args.at == "both":
        baseline_results = {}
        at_results = {}
        for target in targets:
            baseline_results[target] = evaluate_target(
                target, adv_files, X_test, y_test, feature_names,
                at=False, device=args.device, **defense_kwargs,
            )
            at_results[target] = evaluate_target(
                target, adv_files, X_test, y_test, feature_names,
                at=True, device=args.device, **defense_kwargs,
            )
        print_comparison_all(baseline_results, at_results)
    else:
        use_at = args.at == "true"
        all_results = {}
        for target in targets:
            all_results[target] = evaluate_target(
                target, adv_files, X_test, y_test, feature_names,
                at=use_at, device=args.device, **defense_kwargs,
            )
        print_summary(all_results, title=f"Surrogate ResDNN → Targets {'(AT)' if use_at else '(Baseline)'}")


if __name__ == "__main__":
    main()
