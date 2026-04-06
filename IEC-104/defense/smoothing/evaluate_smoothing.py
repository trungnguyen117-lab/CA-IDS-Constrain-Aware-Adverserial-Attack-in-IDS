"""Evaluate Randomized Smoothing defense on adversarial samples.

Compares: No Defense | AT | RS | AT + RS.

RS wraps the prediction process (noise + majority vote), so unlike
purification defenses there are no pre-generated purified CSVs.
Evaluation runs RS at inference time on both clean and adversarial data.

Usage:
    python defense/smoothing/evaluate_smoothing.py --target all --device cpu
    python defense/smoothing/evaluate_smoothing.py --target lstm resdnn --sigma 0.1
    python defense/smoothing/evaluate_smoothing.py --target all --sweep 0.01 0.05 0.1 0.2
"""

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd
import yaml

_HERE = os.path.dirname(os.path.abspath(__file__))
_IEC = os.path.dirname(os.path.dirname(_HERE))
_CLI = os.path.join(_IEC, "cli")
sys.path.insert(0, _IEC)
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_IEC, "script"))
sys.path.insert(0, _CLI)

from smoothing import SmoothedClassifier
from utils.logging import setup_logging, get_logger
from utils.constants import DL_TARGETS, ALL_TARGETS, TRANSFER_SOURCES, ATTACK_COMPAT, WB_ATTACKS
from utils.evaluation import report_metrics, compute_asr
from utils.loaders import load_dataset
from utils.paths import model_path, adv_eval_dir
from model import MODEL_REGISTRY

logger = get_logger(__name__)

N_CLASSES = 12  # IEC-104: 12 classes


def _load_model(target, at=False, device="cpu"):
    suffix = "_at" if at else ""
    ext = ".pth" if target in DL_TARGETS else ".pkl"
    path = model_path(f"framework_{target}_TVAE{suffix}{ext}", at=at)
    cls = MODEL_REGISTRY[target]
    return cls.load(path, device=device)


def _find_adv_csvs(target):
    """Find raw adversarial CSVs for a target."""
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


def evaluate_with_smoothing(target, n_samples, sigma, alpha, feature_std,
                            at=False, device="cpu"):
    """Evaluate a target with RS applied at inference time."""
    tag = f"{target.upper()} {'(AT+RS)' if at else '(RS)'} σ={sigma}"
    logger.info(f"{'='*60}")
    logger.info(f"Evaluating: {tag}  n={n_samples} alpha={alpha}")
    logger.info(f"{'='*60}")

    _, X_test, y_test, feature_names = load_dataset("test")
    m = _load_model(target, at=at, device=device)
    sc = SmoothedClassifier(m, n_samples=n_samples, sigma=sigma,
                            feature_std=feature_std, alpha=alpha,
                            n_classes=N_CLASSES)

    # Clean accuracy with RS
    y_rs = sc.predict(X_test)
    # For metrics: treat ABSTAIN as wrong prediction (class -1 won't match)
    clean_metrics = report_metrics(f"{tag} clean", y_test, y_rs)
    abstain_clean = (y_rs == SmoothedClassifier.ABSTAIN).sum()
    clean_metrics["abstain_rate"] = abstain_clean / len(y_test) * 100
    logger.info(f"{'':>25s}  Abstain={abstain_clean}/{len(y_test)} "
                f"({clean_metrics['abstain_rate']:.1f}%)")
    results = {"clean": clean_metrics}

    # Also record unsmoothed clean accuracy for reference
    y_clean_raw = m.predict(X_test)
    raw_metrics = report_metrics(f"{tag} clean_raw", y_test, y_clean_raw)
    results["clean_raw"] = raw_metrics

    # Adversarial predictions with RS
    adv_files = _find_adv_csvs(target)
    if not adv_files:
        logger.info("No adversarial samples found.")
        return results

    for name, path in adv_files.items():
        try:
            df_adv = pd.read_csv(path)
            X_adv = df_adv[feature_names].values.astype(np.float32)
            y_adv_true = df_adv["Label"].values.astype(int)

            y_adv_rs = sc.predict(X_adv)
            metrics = report_metrics(f"{tag} {name}", y_adv_true, y_adv_rs)

            # ASR: compare with unsmoothed clean predictions
            asr = compute_asr(y_test, y_clean_raw, y_adv_rs)
            metrics["asr"] = asr

            abstain_adv = (y_adv_rs == SmoothedClassifier.ABSTAIN).sum()
            metrics["abstain_rate"] = abstain_adv / len(y_adv_true) * 100

            logger.info(f"{'':>25s}  ASR={asr:6.2f}%  "
                        f"Abstain={abstain_adv}/{len(y_adv_true)}")
            results[name] = metrics
        except Exception as e:
            logger.error(f"Failed evaluating {name}: {e}")

    return results


def evaluate_no_defense(target, device):
    """Evaluate baseline model on raw adversarial data (no defense)."""
    tag = f"{target.upper()} (no defense)"
    _, X_test, y_test, feature_names = load_dataset("test")
    m = _load_model(target, at=False, device=device)
    y_clean = m.predict(X_test)
    clean_metrics = report_metrics(f"{tag} clean", y_test, y_clean)
    results = {"clean": clean_metrics}

    for name, path in _find_adv_csvs(target).items():
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

    for name, path in _find_adv_csvs(target).items():
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


def _print_four_way_comparison(baseline, at, rs, at_rs, sigma):
    """Print 4-way comparison table: No Defense | AT | RS | AT+RS."""
    from rich.table import Table
    from rich.console import Console

    targets = list(baseline.keys())

    # Collect all attack names
    all_attacks = []
    for results_dict in [baseline, at, rs, at_rs]:
        for tgt_results in results_dict.values():
            for name in tgt_results:
                if name not in all_attacks:
                    all_attacks.append(name)

    for target in targets:
        t = Table(
            title=f"{target.upper()} — Randomized Smoothing (σ={sigma})",
            show_lines=True, title_style="bold cyan",
        )
        t.add_column("Attack", style="bold", min_width=18)
        t.add_column("No Defense", justify="right", min_width=12)
        t.add_column("AT", justify="right", min_width=12)
        t.add_column("RS", justify="right", min_width=12)
        t.add_column("AT + RS", justify="right", min_width=12)

        def _fmt(val):
            return f"{val:.2f}%" if val is not None else "-"

        # Clean accuracy
        row = ["Acc (clean)"]
        for rd in [baseline, at, rs, at_rs]:
            row.append(_fmt(rd.get(target, {}).get("clean", {}).get("acc")))
        t.add_row(*row)

        # Abstain rate for RS columns
        rs_abstain = rs.get(target, {}).get("clean", {}).get("abstain_rate")
        at_rs_abstain = at_rs.get(target, {}).get("clean", {}).get("abstain_rate")
        if rs_abstain is not None or at_rs_abstain is not None:
            row = ["Abstain (clean)"]
            row.append("-")
            row.append("-")
            row.append(_fmt(rs_abstain))
            row.append(_fmt(at_rs_abstain))
            t.add_row(*row)

        t.add_section()

        # ASR per attack
        for atk in all_attacks:
            if atk in ("clean", "clean_raw", "clean_purified"):
                continue
            row = [f"ASR ({atk})"]
            for rd in [baseline, at, rs, at_rs]:
                row.append(_fmt(rd.get(target, {}).get(atk, {}).get("asr")))
            t.add_row(*row)

        t.add_section()

        # Accuracy per attack
        for atk in all_attacks:
            if atk in ("clean", "clean_raw", "clean_purified"):
                continue
            row = [f"Acc ({atk})"]
            for rd in [baseline, at, rs, at_rs]:
                row.append(_fmt(rd.get(target, {}).get(atk, {}).get("acc")))
            t.add_row(*row)

        Console().print(t)
        Console().print()


def _print_sigma_sweep(sweep_results):
    """Print sigma sweep comparison table."""
    from rich.table import Table
    from rich.console import Console

    for target, sigma_dict in sweep_results.items():
        sigmas = sorted(sigma_dict.keys())
        t = Table(
            title=f"{target.upper()} — RS Sigma Sweep",
            show_lines=True, title_style="bold magenta",
        )
        t.add_column("Metric", style="bold", min_width=18)
        for s in sigmas:
            t.add_column(f"σ={s}", justify="right", min_width=10)

        def _fmt(val):
            return f"{val:.2f}%" if val is not None else "-"

        # Clean accuracy row
        row = ["Acc (clean)"]
        for s in sigmas:
            row.append(_fmt(sigma_dict[s].get("clean", {}).get("acc")))
        t.add_row(*row)

        # Abstain rate row
        row = ["Abstain (clean)"]
        for s in sigmas:
            row.append(_fmt(sigma_dict[s].get("clean", {}).get("abstain_rate")))
        t.add_row(*row)

        t.add_section()

        # Collect attack names
        all_atks = []
        for s_results in sigma_dict.values():
            for name in s_results:
                if name not in all_atks:
                    all_atks.append(name)

        for atk in all_atks:
            if atk in ("clean", "clean_raw"):
                continue
            row = [f"ASR ({atk})"]
            for s in sigmas:
                row.append(_fmt(sigma_dict[s].get(atk, {}).get("asr")))
            t.add_row(*row)

        Console().print(t)
        Console().print()


def main():
    parser = argparse.ArgumentParser(description="Evaluate Randomized Smoothing defense")
    parser.add_argument("--target", "-tg", nargs="+", required=True,
                        choices=ALL_TARGETS + ["all"])
    parser.add_argument("--sigma", type=float, default=None,
                        help="Noise std (overrides config)")
    parser.add_argument("--n-samples", type=int, default=None,
                        help="Number of noisy samples (overrides config)")
    parser.add_argument("--alpha", type=float, default=None,
                        help="Binomial test significance (overrides config). "
                             "Use 0 to disable abstention.")
    parser.add_argument("--sweep", nargs="+", type=float, default=None,
                        help="Sweep multiple sigma values")
    parser.add_argument("--at", default="both", choices=["false", "true", "both"],
                        help="Also evaluate AT models (default: both)")
    parser.add_argument("--device", "-d", default="auto",
                        choices=["cpu", "cuda", "mps", "auto"])
    parser.add_argument("--config", default=os.path.join(_HERE, "config.yaml"))
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

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    sigma = args.sigma or float(cfg["sigma"])
    n_samples = args.n_samples or int(cfg["n_samples"])
    alpha_val = args.alpha if args.alpha is not None else cfg.get("alpha")
    if alpha_val == 0:
        alpha_val = None  # disable abstention
    elif alpha_val is not None:
        alpha_val = float(alpha_val)

    targets = ALL_TARGETS if "all" in args.target else args.target

    # Compute per-feature std from training data for relative noise scaling
    logger.info("Computing feature std from training data...")
    _, X_train, _, _ = load_dataset("train_tvae")
    feature_std = X_train.std(axis=0).astype(np.float32)
    logger.info(f"Feature std range: [{feature_std.min():.4f}, {feature_std.max():.4f}]")

    # Sigma sweep mode
    if args.sweep:
        sweep_results = {}
        for target in targets:
            sweep_results[target] = {}
            for s in args.sweep:
                logger.info(f"\n--- {target.upper()} σ={s} ---")
                sweep_results[target][s] = evaluate_with_smoothing(
                    target, n_samples, s, alpha_val, feature_std,
                    at=False, device=args.device)
        _print_sigma_sweep(sweep_results)
        return

    # Standard 4-way comparison mode
    baseline_results = {}
    at_results = {}
    rs_results = {}
    at_rs_results = {}

    for target in targets:
        logger.info(f"\n{'#'*60}")
        logger.info(f"# Target: {target.upper()}")
        logger.info(f"{'#'*60}")

        # 1. No defense
        baseline_results[target] = evaluate_no_defense(target, args.device)

        # 2. AT only
        if args.at in ("true", "both"):
            try:
                at_results[target] = evaluate_at_only(target, args.device)
            except Exception as e:
                logger.warning(f"AT model not available for {target}: {e}")
                at_results[target] = {}

        # 3. RS (baseline model + smoothing)
        rs_results[target] = evaluate_with_smoothing(
            target, n_samples, sigma, alpha_val, feature_std,
            at=False, device=args.device)

        # 4. AT + RS (AT model + smoothing)
        if args.at in ("true", "both"):
            try:
                at_rs_results[target] = evaluate_with_smoothing(
                    target, n_samples, sigma, alpha_val, feature_std,
                    at=True, device=args.device)
            except Exception as e:
                logger.warning(f"AT+RS not available for {target}: {e}")
                at_rs_results[target] = {}

    _print_four_way_comparison(baseline_results, at_results, rs_results,
                               at_rs_results, sigma)


if __name__ == "__main__":
    main()
