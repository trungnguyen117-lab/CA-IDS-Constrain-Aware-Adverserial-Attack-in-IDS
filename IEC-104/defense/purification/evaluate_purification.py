"""Evaluate models on purified adversarial samples.

Compares: No Defense | AT | Purification | AT + Purification.

Usage:
    python defense/purification/evaluate_purification.py --target all --device cpu --t 50
    python defense/purification/evaluate_purification.py --target lstm resdnn --at both --t 50
"""

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_IEC = os.path.dirname(os.path.dirname(_HERE))
_CLI = os.path.join(_IEC, "cli")
sys.path.insert(0, _IEC)
sys.path.insert(0, os.path.join(_IEC, "script"))
sys.path.insert(0, _CLI)

from utils.logging import setup_logging, get_logger
from utils.constants import DL_TARGETS, ALL_TARGETS, TRANSFER_SOURCES, ATTACK_COMPAT, WB_ATTACKS
from utils.evaluation import report_metrics, compute_asr
from utils.loaders import load_dataset
from utils.paths import model_path
from model import MODEL_REGISTRY

logger = get_logger(__name__)


def _load_model(target, at=False, device="cpu"):
    suffix = "_at" if at else ""
    ext = ".pkl" if target in ("cat", "rf") else ".pth"
    path = model_path(f"framework_{target}_TVAE{suffix}{ext}", at=at)
    cls = MODEL_REGISTRY[target]
    return cls.load(path, device=device)


def _find_purified_csvs(target, purified_root):
    """Find purified adversarial CSVs for a target (mirrors 2_evaluate._find_adv_csvs logic)."""
    results = {}

    if target in DL_TARGETS:
        d = os.path.join(purified_root, f"{target}_sc")
        if os.path.isdir(d):
            for f in sorted(glob.glob(os.path.join(d, "*_adv.csv"))):
                basename = os.path.basename(f).replace("_adv.csv", "")
                if "surrogate" in basename:
                    continue
                prefix = target + "_"
                atk_name = basename[len(prefix):] if basename.startswith(prefix) else basename
                results[atk_name] = f
    else:
        compatible = ATTACK_COMPAT[target]
        d = os.path.join(purified_root, target)
        if os.path.isdir(d):
            for f in sorted(glob.glob(os.path.join(d, "*_adv.csv"))):
                basename = os.path.basename(f).replace("_adv.csv", "")
                if "surrogate" in basename:
                    continue
                prefix = target + "_"
                atk_name = basename[len(prefix):] if basename.startswith(prefix) else basename
                if atk_name not in compatible:
                    continue
                results[atk_name] = f

        if target in TRANSFER_SOURCES:
            for src in TRANSFER_SOURCES[target]:
                d = os.path.join(purified_root, src)
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


def evaluate_target(target, purified_root, at=False, device="cpu"):
    """Evaluate a target on clean + purified adversarial data."""
    tag = f"{target.upper()} {'(AT+Purif)' if at else '(Purif)'}"
    logger.info(f"{'='*60}")
    logger.info(f"Evaluating: {tag}")
    logger.info(f"{'='*60}")

    _, X_test, y_test, feature_names = load_dataset("test")
    m = _load_model(target, at=at, device=device)

    # Clean accuracy (on unpurified test data, for reference)
    y_clean = m.predict(X_test)
    clean_metrics = report_metrics(f"{tag} clean", y_test, y_clean)
    results = {"clean": clean_metrics}

    # Check for purified clean test (to measure purification impact on clean data)
    purified_clean_path = os.path.join(purified_root, "clean_test_purified.csv")
    if os.path.isfile(purified_clean_path):
        df_pc = pd.read_csv(purified_clean_path)
        X_pc = df_pc[feature_names].values.astype(np.float32)
        y_pc_pred = m.predict(X_pc)
        pc_metrics = report_metrics(f"{tag} clean_purified", y_test, y_pc_pred)
        results["clean_purified"] = pc_metrics

    # Purified adversarial predictions
    adv_files = _find_purified_csvs(target, purified_root)
    if not adv_files:
        logger.info("No purified adversarial samples found.")
        return results

    for name, path in adv_files.items():
        try:
            df_adv = pd.read_csv(path)
            X_adv = df_adv[feature_names].values.astype(np.float32)
            y_adv_true = df_adv["Label"].values.astype(int)

            y_adv_pred = m.predict(X_adv)
            metrics = report_metrics(f"{tag} {name}", y_adv_true, y_adv_pred)

            asr = compute_asr(y_test, y_clean, y_adv_pred)
            metrics["asr"] = asr
            logger.info(f"{'':>25s}  ASR={asr:6.2f}%")
            results[name] = metrics
        except Exception as e:
            logger.error(f"Failed evaluating {name}: {e}")

    return results


def _print_four_way_comparison(baseline, at, purif, at_purif):
    """Print 4-way comparison table: No Defense | AT | Purification | AT+Purification."""
    from rich.table import Table
    from rich.console import Console

    targets = list(baseline.keys())

    # Collect all attack names
    all_attacks = []
    for results_dict in [baseline, at, purif, at_purif]:
        for tgt_results in results_dict.values():
            for name in tgt_results:
                if name not in all_attacks:
                    all_attacks.append(name)

    for target in targets:
        t = Table(
            title=f"{target.upper()} — Defense Comparison",
            show_lines=True, title_style="bold cyan",
        )
        t.add_column("Attack", style="bold", min_width=18)
        t.add_column("No Defense", justify="right", min_width=12)
        t.add_column("AT", justify="right", min_width=12)
        t.add_column("Purification", justify="right", min_width=12)
        t.add_column("AT + Purif", justify="right", min_width=12)

        def _fmt(val):
            return f"{val:.2f}%" if val is not None else "-"

        # Clean accuracy
        row = ["Acc (clean)"]
        for rd in [baseline, at, purif, at_purif]:
            row.append(_fmt(rd.get(target, {}).get("clean", {}).get("acc")))
        t.add_row(*row)

        # Clean purified accuracy (only for purif columns)
        if purif.get(target, {}).get("clean_purified"):
            row = ["Acc (clean purified)"]
            row.append("-")  # baseline
            row.append("-")  # AT
            row.append(_fmt(purif[target].get("clean_purified", {}).get("acc")))
            row.append(_fmt(at_purif.get(target, {}).get("clean_purified", {}).get("acc")))
            t.add_row(*row)

        t.add_section()

        # ASR per attack
        for atk in all_attacks:
            if atk in ("clean", "clean_purified"):
                continue
            row = [f"ASR ({atk})"]
            for rd in [baseline, at, purif, at_purif]:
                row.append(_fmt(rd.get(target, {}).get(atk, {}).get("asr")))
            t.add_row(*row)

        t.add_section()

        # Accuracy per attack
        for atk in all_attacks:
            if atk in ("clean", "clean_purified"):
                continue
            row = [f"Acc ({atk})"]
            for rd in [baseline, at, purif, at_purif]:
                row.append(_fmt(rd.get(target, {}).get(atk, {}).get("acc")))
            t.add_row(*row)

        Console().print(t)
        Console().print()


def main():
    parser = argparse.ArgumentParser(description="Evaluate purification defense")
    parser.add_argument("--target", "-tg", nargs="+", required=True,
                        choices=ALL_TARGETS + ["all"])
    parser.add_argument("--t", type=int, required=True,
                        help="Purification timestep (matches purified/ directory)")
    parser.add_argument("--at", default="both", choices=["false", "true", "both"],
                        help="Also evaluate AT models (default: both)")
    parser.add_argument("--device", "-d", default="auto", choices=["cpu", "cuda", "mps", "auto"])
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    setup_logging(args.log_level)

    import torch
    if args.device == "auto":
        if torch.cuda.is_available():
            args.device = "cuda"
        elif torch.backends.mps.is_available():
            args.device = "mps"
        else:
            args.device = "cpu"

    targets = ALL_TARGETS if "all" in args.target else args.target
    purified_root = os.path.join(_HERE, "purified", f"t_{args.t}")

    if not os.path.isdir(purified_root):
        logger.error(f"Purified directory not found: {purified_root}")
        logger.error("Run purify.py first to generate purified CSVs.")
        sys.exit(1)

    # Evaluate baseline (no defense) on raw adversarial — import from 2_evaluate
    sys.path.insert(0, os.path.join(_CLI, "pipeline"))
    from importlib import import_module

    # We re-implement the baseline evaluation inline to avoid import issues
    from utils.paths import adv_eval_dir

    def _find_adv_csvs_inline(target):
        """Simplified version of 2_evaluate._find_adv_csvs."""
        results = {}
        if target in DL_TARGETS:
            d = adv_eval_dir(f"{target}_sc")
            if os.path.isdir(d):
                for f in sorted(glob.glob(os.path.join(d, "*_adv.csv"))):
                    basename = os.path.basename(f).replace("_adv.csv", "")
                    if "surrogate" in basename:
                        continue
                    prefix = target + "_"
                    atk_name = basename[len(prefix):] if basename.startswith(prefix) else basename
                    results[atk_name] = f
        else:
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
                    results[atk_name] = f
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

    def evaluate_no_defense(target, device):
        """Evaluate baseline model on raw adversarial data (no defense)."""
        tag = f"{target.upper()} (no defense)"
        _, X_test, y_test, feature_names = load_dataset("test")
        m = _load_model(target, at=False, device=device)
        y_clean = m.predict(X_test)
        clean_metrics = report_metrics(f"{tag} clean", y_test, y_clean)
        results = {"clean": clean_metrics}

        for name, path in _find_adv_csvs_inline(target).items():
            try:
                df = pd.read_csv(path)
                X_adv = df[feature_names].values.astype(np.float32)
                y_adv_true = df["Label"].values.astype(int)
                y_adv_pred = m.predict(X_adv)
                metrics = report_metrics(f"{tag} {name}", y_adv_true, y_adv_pred)
                asr = compute_asr(y_test, y_clean, y_adv_pred)
                metrics["asr"] = asr
                results[name] = metrics
            except Exception as e:
                logger.error(f"Failed: {name}: {e}")
        return results

    def evaluate_at_only(target, device):
        """Evaluate AT model on raw adversarial data."""
        tag = f"{target.upper()} (AT)"
        _, X_test, y_test, feature_names = load_dataset("test")
        m = _load_model(target, at=True, device=device)
        y_clean = m.predict(X_test)
        clean_metrics = report_metrics(f"{tag} clean", y_test, y_clean)
        results = {"clean": clean_metrics}

        for name, path in _find_adv_csvs_inline(target).items():
            try:
                df = pd.read_csv(path)
                X_adv = df[feature_names].values.astype(np.float32)
                y_adv_true = df["Label"].values.astype(int)
                y_adv_pred = m.predict(X_adv)
                metrics = report_metrics(f"{tag} {name}", y_adv_true, y_adv_pred)
                asr = compute_asr(y_test, y_clean, y_adv_pred)
                metrics["asr"] = asr
                results[name] = metrics
            except Exception as e:
                logger.error(f"Failed: {name}: {e}")
        return results

    # Collect results
    baseline_results = {}
    at_results = {}
    purif_results = {}
    at_purif_results = {}

    for target in targets:
        logger.info(f"\n{'#'*60}")
        logger.info(f"# Target: {target.upper()}")
        logger.info(f"{'#'*60}")

        # 1. No defense (baseline on raw adv)
        baseline_results[target] = evaluate_no_defense(target, args.device)

        # 2. AT only (AT model on raw adv)
        if args.at in ("true", "both"):
            try:
                at_results[target] = evaluate_at_only(target, args.device)
            except Exception as e:
                logger.warning(f"AT model not available for {target}: {e}")
                at_results[target] = {}

        # 3. Purification (baseline model on purified adv)
        purif_results[target] = evaluate_target(
            target, purified_root, at=False, device=args.device)

        # 4. AT + Purification (AT model on purified adv)
        if args.at in ("true", "both"):
            try:
                at_purif_results[target] = evaluate_target(
                    target, purified_root, at=True, device=args.device)
            except Exception as e:
                logger.warning(f"AT+Purif not available for {target}: {e}")
                at_purif_results[target] = {}

    # Print comparison
    _print_four_way_comparison(baseline_results, at_results, purif_results, at_purif_results)


if __name__ == "__main__":
    main()
