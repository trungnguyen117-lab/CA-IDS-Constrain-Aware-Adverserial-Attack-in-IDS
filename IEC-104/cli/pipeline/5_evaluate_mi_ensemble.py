"""Evaluate MI Adaptive Ensemble on clean + adversarial data.

Same model/CSV loading as 4_evaluate_ensemble.py, but uses
mi_adaptive_voting() instead of weighted_soft_voting().

Supports --mode flag to switch between static and MI ensemble
for direct A/B comparison.
"""

import argparse
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
from utils.constants import (
    ALL_TARGETS, DL_TARGETS, TREE_TARGETS,
    BB_ATTACKS, WB_ATTACKS, ATTACK_COMPAT, TRANSFER_SOURCES,
)
from utils.evaluation import (
    report_metrics, compute_asr, print_summary, print_comparison_all,
)
from utils.loaders import load_dataset
from utils.paths import model_path, adv_eval_dir, set_version, model_stem
from utils.ensemble import weighted_soft_voting, DEFAULT_WEIGHTS
from utils.mi_ensemble import mi_adaptive_voting, MI_DEFAULTS, ADV_WEIGHTS
from utils.defense import create_preprocessing_defense
from model import MODEL_REGISTRY

logger = get_logger(__name__)


# ── Model Loading (same as 4_evaluate_ensemble.py) ──────────────────────────


def _load_model(target, at=False, device="cpu"):
    ext = ".pkl" if target in TREE_TARGETS else ".pth"
    path = model_path(f"{model_stem(target, at=at)}{ext}", at=at)
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


# ── Per-model adversarial CSV mapping (same as 4_evaluate_ensemble.py) ──────


def _adv_csv_path(target, attack):
    if target in DL_TARGETS:
        return adv_eval_dir(f"{target}_sc", f"{target}_{attack}_adv.csv")
    else:
        if attack in BB_ATTACKS:
            return adv_eval_dir(target, f"{target}_{attack}_adv.csv")
        else:
            return adv_eval_dir("resdnn_sc", f"resdnn_{attack}_adv.csv")


def _discover_attacks():
    attacks = []
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

    for atk in sorted(candidate_attacks):
        all_exist = True
        for target in ALL_TARGETS:
            csv_path = _adv_csv_path(target, atk)
            if not os.path.isfile(csv_path):
                all_exist = False
                break
        if all_exist:
            attacks.append(atk)

    return attacks


# ── Voting dispatch ─────────────────────────────────────────────────────────


def _vote(mode, probas_dict, mi_params=None):
    """Dispatch to static or MI voting."""
    if mode == "mi":
        return mi_adaptive_voting(
            probas_dict, base_weights=ADV_WEIGHTS, mi_params=mi_params)
    else:
        return weighted_soft_voting(probas_dict)


# ── Ensemble Evaluation ──────────────────────────────────────────────────────


def evaluate_ensemble(at=False, device="cpu", mode="mi", mi_params=None,
                      defense=None, sigma=0.01):
    tag = f"{'MI' if mode == 'mi' else 'STATIC'} ENSEMBLE ({'AT' if at else 'baseline'})"
    if defense:
        tag += f" +{defense}(σ={sigma})"
    logger.info(f"{'='*60}")
    logger.info(f"Evaluating: {tag}")
    if mode == "mi":
        params = {**MI_DEFAULTS, **(mi_params or {})}
        logger.info(f"MI params: alpha={params['alpha']}, beta={params['beta']}, "
                     f"threshold={params['agreement_threshold']}")
    active_weights = ADV_WEIGHTS if mode == "mi" else DEFAULT_WEIGHTS
    logger.info(f"Base weights: {active_weights}")
    logger.info(f"{'='*60}")

    _, X_test, y_test, feature_names = load_dataset("test")

    models = _load_all_models(at=at, device=device)
    if len(models) < len(ALL_TARGETS):
        logger.error(f"Need all {len(ALL_TARGETS)} models, got {len(models)}")
        return {}

    preprocessing_defences = create_preprocessing_defense(defense, sigma=sigma)
    art_clfs = {}
    for name, m in models.items():
        art_clfs[name] = m.wrap_for_art(
            X_test, preprocessing_defences=preprocessing_defences, device=device,
        )
    proba_fn = lambda m_name, X: models[m_name].art_predict_proba(art_clfs[m_name], X)
    if defense:
        logger.info(f"Defense active: {defense} (sigma={sigma})")

    # --- Clean evaluation ---
    clean_probas = {}
    for name in models:
        clean_probas[name] = proba_fn(name, X_test)

    _, y_ens_clean = _vote(mode, clean_probas, mi_params=mi_params)
    clean_metrics = report_metrics(f"{tag} clean", y_test, y_ens_clean)
    results = {"clean": clean_metrics}

    # --- Adversarial evaluation ---
    attacks = _discover_attacks()
    if not attacks:
        logger.info("No adversarial attacks found for ensemble evaluation.")
        return results

    logger.info(f"Found {len(attacks)} attacks with full coverage: {attacks}")

    for atk in attacks:
        try:
            adv_probas = {}
            for target in models:
                csv_path = _adv_csv_path(target, atk)
                df_adv = pd.read_csv(csv_path)
                X_adv = df_adv[feature_names].values.astype(np.float32)
                adv_probas[target] = proba_fn(target, X_adv)

            _, y_ens_adv = _vote(mode, adv_probas, mi_params=mi_params)

            metrics = report_metrics(f"{tag} {atk}", y_test, y_ens_adv)
            asr = compute_asr(y_test, y_ens_clean, y_ens_adv)
            metrics["asr"] = asr
            logger.info(f"{'':>25s}  ASR={asr:6.2f}%")
            results[atk] = metrics

        except Exception as e:
            logger.error(f"Failed evaluating ensemble on {atk}: {e}")

    return results


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate MI/Static ensemble on clean + adv data")
    parser.add_argument("--at", default="false", choices=["false", "true", "both"],
                        help="Evaluate: false=baseline, true=AT, both=compare")
    parser.add_argument("--mode", "-m", default="mi", choices=["mi", "static"],
                        help="Ensemble mode: mi=MI adaptive, static=weighted soft voting")
    parser.add_argument("--device", "-d", default="cpu",
                        choices=["cpu", "cuda", "auto"])
    parser.add_argument("--defense", default=None,
                        choices=["gaussian_noise", "feature_squeezing"],
                        help="Preprocessing defense")
    parser.add_argument("--sigma", type=float, default=0.01)
    # MI hyperparameters
    parser.add_argument("--alpha", type=float, default=MI_DEFAULTS["alpha"],
                        help=f"MI confidence weight (default: {MI_DEFAULTS['alpha']})")
    parser.add_argument("--beta", type=float, default=MI_DEFAULTS["beta"],
                        help=f"MI disagreement weight (default: {MI_DEFAULTS['beta']})")
    parser.add_argument("--agreement-threshold", type=float,
                        default=MI_DEFAULTS["agreement_threshold"],
                        help=f"MI agreement threshold (default: {MI_DEFAULTS['agreement_threshold']})")
    parser.add_argument("--robustness-bias", type=float,
                        default=MI_DEFAULTS.get("robustness_bias", 0.0),
                        help="Bias toward GBT group on disagreement")
    parser.add_argument("--confidence-gate", type=float,
                        default=MI_DEFAULTS.get("confidence_gate", 0.0),
                        help="Margin threshold for GBT fallback (0=disabled)")
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

    mi_params = {
        "alpha": args.alpha,
        "beta": args.beta,
        "agreement_threshold": args.agreement_threshold,
        "robustness_bias": args.robustness_bias,
        "confidence_gate": args.confidence_gate,
    }
    defense_kwargs = dict(defense=args.defense, sigma=args.sigma)

    if args.at == "both":
        baseline = {"ensemble": evaluate_ensemble(
            at=False, device=args.device, mode=args.mode,
            mi_params=mi_params, **defense_kwargs)}
        at = {"ensemble": evaluate_ensemble(
            at=True, device=args.device, mode=args.mode,
            mi_params=mi_params, **defense_kwargs)}
        print_comparison_all(baseline, at)
        return {"baseline": baseline, "at": at}

    use_at = args.at == "true"
    results = {"ensemble": evaluate_ensemble(
        at=use_at, device=args.device, mode=args.mode,
        mi_params=mi_params, **defense_kwargs)}
    print_summary(results)
    return results


if __name__ == "__main__":
    main()
