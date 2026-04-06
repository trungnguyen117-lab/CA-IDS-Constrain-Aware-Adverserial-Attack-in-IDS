"""Evaluate models on clean test + adversarial samples. Compute ASR."""

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

# Path bootstrap
_HERE = os.path.dirname(os.path.abspath(__file__))
_CLI = os.path.dirname(_HERE)
_MODBUS = os.path.dirname(_CLI)
sys.path.insert(0, _MODBUS)
sys.path.insert(0, _CLI)

from utils.logging import setup_logging, get_logger
from utils.constants import (
    DL_TARGETS, TREE_TARGETS, ALL_TARGETS,
    TRANSFER_SOURCES, ATTACK_COMPAT, WB_ATTACKS,
)
from utils.evaluation import report_metrics, compute_asr, print_summary, print_comparison_all
from utils.loaders import load_dataset, load_clip_values
from utils.paths import model_path, adv_eval_dir
from utils.defense import create_preprocessing_defense
from model import MODEL_REGISTRY

logger = get_logger(__name__)


def _load_model(target, at=False, device="cpu"):
    suffix = "_at" if at else ""
    ext = ".pth" if target in DL_TARGETS else ".pkl"
    path = model_path(f"framework_{target}_TabDiff{suffix}{ext}", at=at)
    return MODEL_REGISTRY[target].load(path, device=device)


def _find_adv_csvs(target):
    """Find adversarial eval CSVs for a target."""
    results = {}

    if target in DL_TARGETS:
        d = adv_eval_dir(f"{target}_sc")
        if os.path.isdir(d):
            for f in sorted(glob.glob(os.path.join(d, "*_adv.csv"))):
                basename = os.path.basename(f).replace("_adv.csv", "")
                prefix = target + "_"
                atk_name = basename[len(prefix):] if basename.startswith(prefix) else basename
                results[atk_name] = f
    else:
        # Tree: own direct BB attacks
        compatible = ATTACK_COMPAT[target]
        d = adv_eval_dir(target)
        if os.path.isdir(d):
            for f in sorted(glob.glob(os.path.join(d, "*_adv.csv"))):
                basename = os.path.basename(f).replace("_adv.csv", "")
                prefix = target + "_"
                atk_name = basename[len(prefix):] if basename.startswith(prefix) else basename
                if atk_name not in compatible:
                    continue
                results[atk_name] = f

        # Tree: transfer WB from DL (ftt_sc)
        if target in TRANSFER_SOURCES:
            for src in TRANSFER_SOURCES[target]:
                d = adv_eval_dir(src)
                if os.path.isdir(d):
                    for f in sorted(glob.glob(os.path.join(d, "*_adv.csv"))):
                        basename = os.path.basename(f).replace("_adv.csv", "")
                        src_model = src.replace("_sc", "")
                        prefix = src_model + "_"
                        atk_name = basename[len(prefix):] if basename.startswith(prefix) else basename
                        if atk_name not in WB_ATTACKS:
                            continue
                        results[f"transfer_{src_model}_{atk_name}"] = f

    return results


def evaluate_target(target, at=False, device="cpu", defense=None, sigma=0.01,
                    clip_values=None):
    logger.info(f"{'='*60}")
    tag = f"{target.upper()} {'(AT)' if at else '(baseline)'}"
    if defense:
        tag += f" +{defense}(σ={sigma})"
    logger.info(f"Evaluating: {tag}")
    logger.info(f"{'='*60}")

    _, X_test, y_test, feature_names = load_dataset("test")
    m = _load_model(target, at=at, device=device)

    preprocessing_defences = create_preprocessing_defense(defense, sigma=sigma)
    art_clf = m.wrap_for_art(
        X_test, clip_values=clip_values,
        preprocessing_defences=preprocessing_defences, device=device,
    )
    predict_fn = lambda X: np.argmax(art_clf.predict(X), axis=1)

    y_clean = predict_fn(X_test)
    clean_metrics = report_metrics(f"{tag} clean", y_test, y_clean)
    results = {"clean": clean_metrics}

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
                        choices=ALL_TARGETS + ["all"])
    parser.add_argument("--at", default="false", choices=["false", "true", "both"])
    parser.add_argument("--device", "-d", default="cpu",
                        choices=["cpu", "cuda", "mps", "auto"])
    parser.add_argument("--defense", default=None,
                        choices=["gaussian_noise", "feature_squeezing"])
    parser.add_argument("--sigma", type=float, default=0.01)
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

    targets = ALL_TARGETS if "all" in args.target else args.target
    defense_kwargs = dict(defense=args.defense, sigma=args.sigma)
    clip_values = load_clip_values()

    if args.at == "both":
        baseline_results, at_results = {}, {}
        for target in targets:
            baseline_results[target] = evaluate_target(
                target, at=False, device=args.device,
                clip_values=clip_values, **defense_kwargs)
            at_results[target] = evaluate_target(
                target, at=True, device=args.device,
                clip_values=clip_values, **defense_kwargs)
        print_comparison_all(baseline_results, at_results)
        return

    use_at = args.at == "true"
    all_results = {}
    for target in targets:
        all_results[target] = evaluate_target(
            target, at=use_at, device=args.device,
            clip_values=clip_values, **defense_kwargs)
    print_summary(all_results)


if __name__ == "__main__":
    main()
