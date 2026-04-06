"""Evaluate DAE Purification + Ensemble defense for IEC-104.

Full encode-decode: adversarial input → DAE → purified input → existing FOAMI
ensemble models predict on purified features. No retraining needed.

4-way: Ensemble | Ensemble+AT | DAE+Ensemble | DAE+Ensemble+AT

Usage:
    python defense/dae/evaluate_dae_ensemble.py --device mps --at both
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

# Path bootstrap
_HERE = os.path.dirname(os.path.abspath(__file__))
_IEC = os.path.dirname(os.path.dirname(_HERE))
_CLI = os.path.join(_IEC, "cli")
sys.path.insert(0, _IEC)
sys.path.insert(0, os.path.join(_IEC, "script"))
sys.path.insert(0, _CLI)

from utils.logging import setup_logging, get_logger
from utils.constants import (
    ALL_TARGETS, DL_TARGETS, TREE_TARGETS,
    BB_ATTACKS, WB_ATTACKS,
)
from utils.evaluation import report_metrics, compute_asr
from utils.loaders import load_dataset
from utils.paths import model_path, adv_eval_dir
from utils.ensemble import weighted_soft_voting, DEFAULT_WEIGHTS
from model import MODEL_REGISTRY

logger = get_logger(__name__)

_PURIFIED_ROOT = os.path.join(_HERE, "purified")


# ── Model Loading ─────────────────────────────────────────────────────────────


def _load_model(target, at=False, device="cpu"):
    suffix = "_at" if at else ""
    ext = ".pkl" if target in TREE_TARGETS else ".pth"
    path = model_path(f"framework_{target}_TVAE{suffix}{ext}", at=at)
    return MODEL_REGISTRY[target].load(path, device=device)


def _load_all_models(at=False, device="cpu"):
    models = {}
    for target in ALL_TARGETS:
        try:
            models[target] = _load_model(target, at=at, device=device)
            logger.info(f"Loaded {target} {'(AT)' if at else '(baseline)'}")
        except Exception as e:
            logger.error(f"Failed loading {target}: {e}")
    return models


# ── CSV path resolution ──────────────────────────────────────────────────────


def _adv_csv_path(target, attack):
    """Raw adversarial CSV path."""
    if target in DL_TARGETS:
        return adv_eval_dir(f"{target}_sc", f"{target}_{attack}_adv.csv")
    else:
        if attack in BB_ATTACKS:
            return adv_eval_dir(target, f"{target}_{attack}_adv.csv")
        else:
            return adv_eval_dir("resdnn_sc", f"resdnn_{attack}_adv.csv")


def _purified_csv_path(target, attack):
    """DAE-purified adversarial CSV path (mirrors adv_eval structure)."""
    if target in DL_TARGETS:
        return os.path.join(
            _PURIFIED_ROOT, f"{target}_sc", f"{target}_{attack}_adv.csv")
    else:
        if attack in BB_ATTACKS:
            return os.path.join(
                _PURIFIED_ROOT, target, f"{target}_{attack}_adv.csv")
        else:
            return os.path.join(
                _PURIFIED_ROOT, "resdnn_sc", f"resdnn_{attack}_adv.csv")


def _discover_attacks():
    """Find attacks that have both raw AND purified CSVs for all targets."""
    candidate_attacks = set()
    for dl_target in DL_TARGETS:
        d = adv_eval_dir(f"{dl_target}_sc")
        if not os.path.isdir(d):
            continue
        for fname in os.listdir(d):
            if not fname.endswith("_adv.csv") or "surrogate" in fname:
                continue
            basename = fname.replace("_adv.csv", "")
            prefix = dl_target + "_"
            atk = basename[len(prefix):] if basename.startswith(prefix) else basename
            candidate_attacks.add(atk)

    for tree_target in TREE_TARGETS:
        d = adv_eval_dir(tree_target)
        if not os.path.isdir(d):
            continue
        for fname in os.listdir(d):
            if not fname.endswith("_adv.csv") or "surrogate" in fname:
                continue
            basename = fname.replace("_adv.csv", "")
            prefix = tree_target + "_"
            atk = basename[len(prefix):] if basename.startswith(prefix) else basename
            candidate_attacks.add(atk)

    attacks = []
    for atk in sorted(candidate_attacks):
        # Check both raw and purified exist for all targets
        all_exist = True
        for target in ALL_TARGETS:
            if not os.path.isfile(_adv_csv_path(target, atk)):
                all_exist = False
                break
            if not os.path.isfile(_purified_csv_path(target, atk)):
                all_exist = False
                break
        if all_exist:
            attacks.append(atk)

    return attacks


# ── Evaluation Core ──────────────────────────────────────────────────────────


def evaluate_ensemble(models, X_test, y_test, feature_names, attacks,
                      purified=False, at=False, device="cpu"):
    """Evaluate ensemble on raw or DAE-purified adversarial data.

    When purified=True, loads purified CSVs instead of raw adversarial CSVs.
    """
    if purified and at:
        variant = "DAE+Ens+AT"
    elif purified:
        variant = "DAE+Ensemble"
    elif at:
        variant = "Ensemble+AT"
    else:
        variant = "Ensemble"

    logger.info(f"{'='*60}")
    logger.info(f"Evaluating: {variant}")
    logger.info(f"{'='*60}")

    # --- Clean evaluation ---
    if purified:
        # Use purified clean test to measure DAE impact on clean data
        purified_clean = os.path.join(_PURIFIED_ROOT, "clean_test_purified.csv")
        if os.path.isfile(purified_clean):
            df_pc = pd.read_csv(purified_clean)
            X_clean_input = df_pc[feature_names].values.astype(np.float32)
            logger.info(f"Using purified clean test: {purified_clean}")
        else:
            logger.warning("Purified clean test not found, using raw test data")
            X_clean_input = X_test
    else:
        X_clean_input = X_test

    clean_probas = {}
    for name, m in models.items():
        art_clf = m.wrap_for_art(X_clean_input, device=device)
        proba = m.art_predict_proba(art_clf, X_clean_input)
        clean_probas[name] = proba

    _, y_ens_clean = weighted_soft_voting(clean_probas)
    clean_metrics = report_metrics(f"{variant} clean", y_test, y_ens_clean)
    results = {"clean": clean_metrics}

    # --- Adversarial evaluation ---
    if not attacks:
        logger.info("No adversarial attacks found.")
        return results

    for atk in attacks:
        try:
            adv_probas = {}
            for target in models:
                if purified:
                    csv_path = _purified_csv_path(target, atk)
                else:
                    csv_path = _adv_csv_path(target, atk)

                df_adv = pd.read_csv(csv_path)
                X_adv = df_adv[feature_names].values.astype(np.float32)

                art_clf = models[target].wrap_for_art(X_adv, device=device)
                proba = models[target].art_predict_proba(art_clf, X_adv)
                adv_probas[target] = proba

            _, y_ens_adv = weighted_soft_voting(adv_probas)

            metrics = report_metrics(f"{variant} {atk}", y_test, y_ens_adv)
            asr = compute_asr(y_test, y_ens_clean, y_ens_adv)
            metrics["asr"] = asr
            logger.info(f"{'':>25s}  ASR={asr:6.2f}%")
            results[atk] = metrics

        except Exception as e:
            logger.error(f"Failed evaluating {variant} on {atk}: {e}")

    return results


# ── Rich Table ───────────────────────────────────────────────────────────────


def _print_four_way(ens_base, ens_at, dae_base, dae_at, attacks):
    from rich.table import Table
    from rich.console import Console

    t = Table(
        title="DAE Purification + Ensemble Defense",
        show_lines=True, title_style="bold cyan",
    )
    t.add_column("Metric", style="bold", min_width=18)
    t.add_column("Ensemble", justify="right", min_width=12)
    t.add_column("Ensemble+AT", justify="right", min_width=12)
    t.add_column("DAE+Ensemble", justify="right", min_width=12)
    t.add_column("DAE+Ens+AT", justify="right", min_width=12)

    def _fmt(val):
        return f"{val:.2f}%" if val is not None else "-"

    for key, label in [("acc", "Acc (clean)"), ("f1", "F1 (clean)")]:
        row = [label]
        for rd in [ens_base, ens_at, dae_base, dae_at]:
            row.append(_fmt(rd.get("clean", {}).get(key)))
        t.add_row(*row)

    t.add_section()

    for atk in attacks:
        row = [f"ASR ({atk})"]
        for rd in [ens_base, ens_at, dae_base, dae_at]:
            row.append(_fmt(rd.get(atk, {}).get("asr")))
        t.add_row(*row)

    t.add_section()

    for atk in attacks:
        row = [f"Acc ({atk})"]
        for rd in [ens_base, ens_at, dae_base, dae_at]:
            row.append(_fmt(rd.get(atk, {}).get("acc")))
        t.add_row(*row)

    Console().print(t)
    Console().print()


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate DAE Purification + Ensemble defense")
    parser.add_argument("--at", default="both",
                        choices=["false", "true", "both"])
    parser.add_argument("--device", "-d", default="auto",
                        choices=["cpu", "cuda", "mps", "auto"])
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
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
    logger.info(f"Device: {args.device}")

    if not os.path.isdir(_PURIFIED_ROOT):
        logger.error(f"Purified directory not found: {_PURIFIED_ROOT}")
        logger.error("Run purify_dae.py first.")
        sys.exit(1)

    _, X_test, y_test, feature_names = load_dataset("test")

    attacks = _discover_attacks()
    if not attacks:
        logger.error("No attacks with both raw + purified CSVs found.")
        sys.exit(1)
    logger.info(f"Found {len(attacks)} attacks: {attacks}")

    # --- 1. Ensemble (baseline) ---
    models_base = _load_all_models(at=False, device=args.device)
    if len(models_base) < len(ALL_TARGETS):
        logger.error("Cannot load all baseline models.")
        sys.exit(1)

    ens_base = evaluate_ensemble(
        models_base, X_test, y_test, feature_names, attacks,
        device=args.device,
    )

    # --- 2. DAE + Ensemble ---
    dae_base = evaluate_ensemble(
        models_base, X_test, y_test, feature_names, attacks,
        purified=True, device=args.device,
    )

    # --- 3 & 4. With AT ---
    ens_at = {}
    dae_at = {}

    if args.at in ("true", "both"):
        models_at = _load_all_models(at=True, device=args.device)
        if len(models_at) >= len(ALL_TARGETS):
            ens_at = evaluate_ensemble(
                models_at, X_test, y_test, feature_names, attacks,
                at=True, device=args.device,
            )
            dae_at = evaluate_ensemble(
                models_at, X_test, y_test, feature_names, attacks,
                purified=True, at=True, device=args.device,
            )
        else:
            logger.warning("Could not load all AT models.")

    _print_four_way(ens_base, ens_at, dae_base, dae_at, attacks)


if __name__ == "__main__":
    main()
