"""Evaluate models with AE-based adversarial detection defense.

Same structure as 6_evaluate_dse.py but uses reconstruction error
instead of query simulation for detection.
"""

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd
import yaml

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
from utils.evaluation import report_metrics, compute_asr
from utils.loaders import load_dataset, load_clip_values
from utils.paths import get_path, model_path, adv_eval_dir, training_config_path
from utils.defense import create_preprocessing_defense
from utils.dse_ae_detector import AEDetector
from model import MODEL_REGISTRY
from model.dse_ae import DSEAutoencoder

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


def evaluate_with_ae(target, detector, at=False, device="cpu",
                     defense=None, sigma=0.01, clip_values=None):
    """Evaluate a target model with AE-based detection."""
    logger.info(f"{'='*60}")
    tag = f"{target.upper()} {'(AT)' if at else '(baseline)'} +AE"
    if defense:
        tag += f" +{defense}(sigma={sigma})"
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

    # FPR on clean data (clean samples falsely rejected)
    fpr_clean = detector.compute_fpr(X_test)
    clean_metrics["ae_fpr"] = fpr_clean
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

            # Standard ASR
            asr = compute_asr(y_test, y_clean, y_adv_pred)
            metrics["asr"] = asr
            logger.info(f"{'':>25s}  ASR={asr:6.2f}%")

            # AE detection + reject
            det_rate = detector.compute_detection_rate(X_adv)
            adj_asr = detector.adjusted_asr(
                y_test, y_clean, y_adv_pred, X_adv,
            )
            metrics["ae_det_rate"] = det_rate
            metrics["ae_adj_asr"] = adj_asr
            logger.info(
                f"{'':>25s}  AE: det={det_rate:.2f}%, "
                f"adj_ASR={adj_asr:.2f}% (was {asr:.2f}%)"
            )

            results[name] = metrics
        except Exception as e:
            logger.error(f"Failed evaluating {name}: {e}")

    return results


def _print_ae_summary(all_results, title="AE DETECTION SUMMARY"):
    """Print summary table with AE detection metrics."""
    from rich.console import Console
    from rich.table import Table

    targets = list(all_results.keys())
    all_attacks = []
    for results in all_results.values():
        for name in results:
            if name != "clean" and name not in all_attacks:
                all_attacks.append(name)

    t = Table(title=title, show_lines=True, title_style="bold cyan")
    t.add_column("Metric", style="bold", min_width=25)
    for tgt in targets:
        t.add_column(tgt.upper(), justify="right", min_width=12)

    # Clean metrics
    for key, label in [("acc", "Acc (clean)"), ("f1", "F1 (clean)")]:
        row = [label]
        for tgt in targets:
            val = all_results[tgt].get("clean", {}).get(key)
            row.append(f"{val:.2f}%" if val is not None else "-")
        t.add_row(*row)

    # Clean FPR
    row = ["FPR (clean, AE)"]
    for tgt in targets:
        val = all_results[tgt].get("clean", {}).get("ae_fpr")
        row.append(f"{val:.2f}%" if val is not None else "-")
    t.add_row(*row)

    t.add_section()

    # Per-attack metrics
    for atk in all_attacks:
        row = [f"ASR ({atk})"]
        for tgt in targets:
            val = all_results[tgt].get(atk, {}).get("asr")
            row.append(f"{val:.2f}%" if val is not None else "-")
        t.add_row(*row)

        row = [f"ASR+AE ({atk})"]
        for tgt in targets:
            val = all_results[tgt].get(atk, {}).get("ae_adj_asr")
            row.append(f"{val:.2f}%" if val is not None else "-")
        t.add_row(*row, style="green")

        row = [f"Det% ({atk})"]
        for tgt in targets:
            val = all_results[tgt].get(atk, {}).get("ae_det_rate")
            row.append(f"{val:.2f}%" if val is not None else "-")
        t.add_row(*row)

    Console().print(t)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate models with AE-based adversarial detection"
    )
    parser.add_argument("--target", "-t", nargs="+", required=True,
                        choices=ALL_TARGETS + ["all"])
    parser.add_argument("--at", default="false", choices=["false", "true"])
    parser.add_argument("--device", "-d", default="cpu",
                        choices=["cpu", "cuda", "mps", "auto"])
    parser.add_argument("--defense", default=None,
                        choices=["gaussian_noise", "feature_squeezing"])
    parser.add_argument("--sigma", type=float, default=0.01)
    parser.add_argument("--ae-model", default=None,
                        help="Path to AE checkpoint")
    parser.add_argument("--calibrate", action="store_true",
                        help="Run threshold calibration sweep")
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

    # Load AE model
    ae_path = args.ae_model or os.path.join(
        get_path("models"), "dse_ae.pth"
    )
    ae = DSEAutoencoder.load(ae_path, device=args.device)

    # Load AE config for detection params
    ae_cfg_path = training_config_path("dse_ae")
    if os.path.isfile(ae_cfg_path):
        with open(ae_cfg_path) as f:
            ae_cfg = yaml.safe_load(f) or {}
    else:
        ae_cfg = {}

    # Build detector from clean training data
    _, X_clean, _, _ = load_dataset("train_tvae")
    detector = AEDetector(ae, X_clean, cfg=ae_cfg)

    # Optional calibration
    if args.calibrate:
        _, X_test, _, feature_names = load_dataset("test")
        for target in ALL_TARGETS:
            adv_files = _find_adv_csvs(target)
            if adv_files:
                first_path = list(adv_files.values())[0]
                df_adv = pd.read_csv(first_path)
                X_adv = df_adv[feature_names].values.astype(np.float32)
                logger.info(f"Calibrating with {list(adv_files.keys())[0]} "
                            f"({len(X_adv)} samples)")
                detector.calibrate_threshold(X_test, X_adv)
                break
        return

    targets = ALL_TARGETS if "all" in args.target else args.target
    use_at = args.at == "true"
    defense_kwargs = dict(defense=args.defense, sigma=args.sigma)
    clip_values = load_clip_values()

    all_results = {}
    for target in targets:
        all_results[target] = evaluate_with_ae(
            target, detector, at=use_at, device=args.device,
            clip_values=clip_values, **defense_kwargs,
        )

    _print_ae_summary(
        all_results,
        title=f"AE DETECTION {'(AT)' if use_at else '(baseline)'}",
    )


if __name__ == "__main__":
    main()
