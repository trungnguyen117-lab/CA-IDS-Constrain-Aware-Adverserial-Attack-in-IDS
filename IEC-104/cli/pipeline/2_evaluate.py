"""Evaluate models on clean test + adversarial samples. Compute ASR.
DL models have InputNorm embedded — no external ScaledModel needed.
All models predict directly on raw data.
"""

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

# Path bootstrap — _CLI must be first so cli/utils/ shadows script/utils/
_HERE = os.path.dirname(os.path.abspath(__file__))
_CLI = os.path.dirname(_HERE)
_IEC = os.path.dirname(_CLI)
sys.path.insert(0, _IEC)
sys.path.insert(0, os.path.join(_IEC, "script"))
sys.path.insert(0, _CLI)

from utils.logging import setup_logging, get_logger
from utils.constants import DL_TARGETS, ALL_TARGETS, TRANSFER_SOURCES, ATTACK_COMPAT
from utils.evaluation import report_metrics, compute_asr, print_summary, print_comparison_all
from utils.loaders import load_dataset
from utils.paths import model_path, adv_eval_dir, set_version, model_stem
from utils.defense import create_preprocessing_defense
from model import get_model, MODEL_REGISTRY

logger = get_logger(__name__)




def _load_model(target, at=False, device="cpu"):
    """Load model via registry. Returns a BaseModel instance."""
    ext = ".pkl" if target in ("cat", "rf") else ".pth"
    path = model_path(f"{model_stem(target, at=at)}{ext}", at=at)
    cls = MODEL_REGISTRY[target]
    return cls.load(path, device=device)


def _find_adv_csvs(target):
    """Find adversarial eval CSVs for a target.

    DL (lstm, resdnn): only adv_eval/{target}_sc/ — own WB+BB in scaled space.
    Tree (cat, rf): adv_eval/{target}/ (own BB: zoo, hsja)
                    + transfer from adv_eval/resdnn_sc/.
    Surrogate files are skipped.
    """
    results = {}

    if target in DL_TARGETS:
        # DL: only scaled-space attacks generated against this model
        d = adv_eval_dir(f"{target}_sc")
        if os.path.isdir(d):
            for f in sorted(glob.glob(os.path.join(d, "*_adv.csv"))):
                basename = os.path.basename(f).replace("_adv.csv", "")
                if "surrogate" in basename:
                    continue
                prefix = target + "_"
                atk_name = basename[len(prefix):] if basename.startswith(prefix) else basename
                results[f"{atk_name}"] = f
    else:
        # Tree: own direct BB attacks only (zoo, hsja)
        compatible = ATTACK_COMPAT[target]
        d = adv_eval_dir(target)
        if os.path.isdir(d):
            for f in sorted(glob.glob(os.path.join(d, "*_adv.csv"))):
                basename = os.path.basename(f).replace("_adv.csv", "")
                if "surrogate" in basename:
                    continue
                prefix = target + "_"
                atk_name = basename[len(prefix):] if basename.startswith(prefix) else basename
                if atk_name not in compatible:
                    continue
                results[f"{atk_name}"] = f

        # Tree: transfer WB-only attacks from DL models (resdnn_sc)
        # Exclude BB (zoo, hsja) — those are evaluated directly on the tree model
        from utils.constants import WB_ATTACKS
        if target in TRANSFER_SOURCES:
            for src in TRANSFER_SOURCES[target]:
                d = adv_eval_dir(src)
                if os.path.isdir(d):
                    for f in sorted(glob.glob(os.path.join(d, "*_adv.csv"))):
                        basename = os.path.basename(f).replace("_adv.csv", "")
                        if "surrogate" in basename:
                            continue
                        src_model = src.replace("_sc", "")
                        prefix = src_model + "_"
                        atk_name = basename[len(prefix):] if basename.startswith(prefix) else basename
                        if atk_name not in WB_ATTACKS:
                            continue
                        results[f"transfer_{src_model}_{atk_name}"] = f

    return results




def evaluate_target(target, at=False, device="cpu", defense=None, sigma=0.01):
    """Evaluate a single target on clean + adversarial data. Returns results dict."""
    logger.info(f"{'='*60}")
    tag = f"{target.upper()} {'(AT)' if at else '(baseline)'}"
    if defense:
        tag += f" +{defense}(σ={sigma})"
    logger.info(f"Evaluating: {tag}")
    logger.info(f"{'='*60}")

    # Load test data via shared loader
    _, X_test, y_test, feature_names = load_dataset("test")
    # Load model via registry
    m = _load_model(target, at=at, device=device)

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

    # Adversarial predictions — load raw CSVs, predict via model
    adv_files = _find_adv_csvs(target)
    if not adv_files:
        logger.info("No adversarial samples found for evaluation.")
        return results

    for name, path in adv_files.items():
        try:
            df_adv = pd.read_csv(path)
            X_adv = df_adv[feature_names].values.astype(np.float32)
            y_adv_true = df_adv["Label"].values.astype(int)

            y_adv_pred = predict_fn(X_adv)
            metrics = report_metrics(f"{tag} {name}", y_adv_true, y_adv_pred)

            # ASR on correctly-classified subset
            asr = compute_asr(y_test, y_clean, y_adv_pred)
            metrics["asr"] = asr
            logger.info(f"{'':>25s}  ASR={asr:6.2f}%")
            results[name] = metrics
        except Exception as e:
            logger.error(f"Failed evaluating {name}: {e}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate models on clean + adv data")
    parser.add_argument("--target", "-t", nargs="+", required=True,
                        choices=ALL_TARGETS + ["all"],
                        help="Models to evaluate")
    parser.add_argument("--at", default="false", choices=["false", "true", "both"],
                        help="Evaluate: false=baseline, true=AT, both=compare")
    parser.add_argument("--device", "-d", default="cpu",
                        choices=["cpu", "cuda", "auto"])
    parser.add_argument("--defense", default=None,
                        choices=["gaussian_noise", "feature_squeezing"],
                        help="Preprocessing defense to apply during inference")
    parser.add_argument("--sigma", type=float, default=0.01,
                        help="Gaussian noise sigma (default: 0.01)")
    parser.add_argument("--version", "-V", default="v1",
                        help="Version tag for adv samples/models (default: v1)")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    setup_logging(args.log_level)
    set_version(args.version)

    if args.device == "auto":
        import torch
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    targets = ALL_TARGETS if "all" in args.target else args.target

    defense_kwargs = dict(defense=args.defense, sigma=args.sigma)

    if args.at == "both":
        baseline_results = {}
        at_results = {}
        for target in targets:
            baseline_results[target] = evaluate_target(
                target, at=False, device=args.device, **defense_kwargs)
            at_results[target] = evaluate_target(
                target, at=True, device=args.device, **defense_kwargs)
        print_comparison_all(baseline_results, at_results)
        return {"baseline": baseline_results, "at": at_results}

    use_at = args.at == "true"
    all_results = {}
    for target in targets:
        all_results[target] = evaluate_target(
            target, at=use_at, device=args.device, **defense_kwargs)
    print_summary(all_results)
    return all_results


if __name__ == "__main__":
    main()
