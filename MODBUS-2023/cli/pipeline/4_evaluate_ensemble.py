"""Evaluate Weighted Soft Voting + MI ensemble on clean + adversarial data."""

import argparse
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
    ALL_TARGETS, DL_TARGETS, TREE_TARGETS,
    BB_ATTACKS, WB_ATTACKS, ATTACK_COMPAT, TRANSFER_SOURCES,
    GBT_GROUP, DL_GROUP, DEFAULT_ENSEMBLE_WEIGHTS,
)
from utils.evaluation import (
    report_metrics, compute_asr, print_summary, print_comparison_all,
)
from utils.loaders import load_dataset, load_clip_values
from utils.paths import model_path, adv_eval_dir
from utils.ensemble import weighted_soft_voting, mutual_inference
from utils.defense import create_preprocessing_defense
from model import MODEL_REGISTRY

logger = get_logger(__name__)


def _load_model(target, at=False, device="cpu"):
    suffix = "_at" if at else ""
    ext = ".pth" if target in DL_TARGETS else ".pkl"
    path = model_path(f"framework_{target}_TabDiff{suffix}{ext}", at=at)
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


def _adv_csv_path(target, attack):
    """Resolve the correct adversarial CSV path for a (target, attack) pair."""
    if target in DL_TARGETS:
        return adv_eval_dir(f"{target}_sc", f"{target}_{attack}_adv.csv")
    else:
        if attack in BB_ATTACKS:
            return adv_eval_dir(target, f"{target}_{attack}_adv.csv")
        else:
            # Transfer WB from ftt_sc
            return adv_eval_dir("ftt_sc", f"ftt_{attack}_adv.csv")


def _discover_attacks():
    """Discover attacks with full coverage across all loaded models."""
    candidate_attacks = set()

    for dl_target in DL_TARGETS:
        d = adv_eval_dir(f"{dl_target}_sc")
        if not os.path.isdir(d):
            continue
        for fname in os.listdir(d):
            if not fname.endswith("_adv.csv"):
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
            if not fname.endswith("_adv.csv"):
                continue
            basename = fname.replace("_adv.csv", "")
            prefix = tree_target + "_"
            atk = basename[len(prefix):] if basename.startswith(prefix) else basename
            candidate_attacks.add(atk)

    attacks = []
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


def evaluate_ensemble(at=False, device="cpu", defense=None, sigma=0.01,
                      clip_values=None):
    tag = "ENSEMBLE (AT)" if at else "ENSEMBLE (baseline)"
    logger.info(f"{'='*60}")
    logger.info(f"Evaluating: {tag}")
    logger.info(f"Weights: {DEFAULT_ENSEMBLE_WEIGHTS}")
    logger.info(f"{'='*60}")

    _, X_test, y_test, feature_names = load_dataset("test")

    models = _load_all_models(at=at, device=device)
    if len(models) < len(ALL_TARGETS):
        logger.error(
            f"Need all {len(ALL_TARGETS)} models for ensemble, got {len(models)}")
        return {}

    preprocessing_defences = create_preprocessing_defense(defense, sigma=sigma)
    art_clfs = {}
    for name, m in models.items():
        art_clfs[name] = m.wrap_for_art(
            X_test, clip_values=clip_values,
            preprocessing_defences=preprocessing_defences, device=device,
        )

    def proba_fn(m_name, X):
        return np.asarray(art_clfs[m_name].predict(X), dtype=np.float64)

    # --- Clean: Weighted Soft Voting ---
    clean_probas = {name: proba_fn(name, X_test) for name in models}
    _, y_ens_clean = weighted_soft_voting(clean_probas)
    clean_metrics = report_metrics(f"{tag} WSV clean", y_test, y_ens_clean)
    results = {"clean": clean_metrics}

    # --- Clean: MI Ensemble ---
    gbt_probas = {n: clean_probas[n] for n in GBT_GROUP if n in models}
    dl_probas = [clean_probas[n] for n in DL_GROUP if n in models]
    if gbt_probas and dl_probas:
        gbt_w = np.array([DEFAULT_ENSEMBLE_WEIGHTS.get(n, 0.0) for n in gbt_probas])
        gbt_w = gbt_w / (gbt_w.sum() + 1e-10)
        # Default MI params (alpha=0.3, beta=1.0, threshold=0.3)
        y_mi, _ = mutual_inference(gbt_probas, dl_probas, gbt_w, 0.3, 1.0, 0.3)
        mi_metrics = report_metrics(f"{tag} MI clean", y_test, y_mi)
        results["clean_mi"] = mi_metrics

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

            # WSV
            _, y_ens_adv = weighted_soft_voting(adv_probas)
            metrics = report_metrics(f"{tag} WSV {atk}", y_test, y_ens_adv)
            asr = compute_asr(y_test, y_ens_clean, y_ens_adv)
            metrics["asr"] = asr
            logger.info(f"{'':>25s}  ASR={asr:6.2f}%")
            results[atk] = metrics

            # MI
            if gbt_probas and dl_probas:
                gbt_adv = {n: adv_probas[n] for n in GBT_GROUP if n in models}
                dl_adv = [adv_probas[n] for n in DL_GROUP if n in models]
                y_mi_adv, _ = mutual_inference(gbt_adv, dl_adv, gbt_w, 0.3, 1.0, 0.3)
                mi_metrics = report_metrics(f"{tag} MI {atk}", y_test, y_mi_adv)
                mi_asr = compute_asr(y_test, y_mi, y_mi_adv)
                mi_metrics["asr"] = mi_asr
                logger.info(f"{'':>25s}  MI ASR={mi_asr:6.2f}%")
                results[f"{atk}_mi"] = mi_metrics

        except Exception as e:
            logger.error(f"Failed evaluating ensemble on {atk}: {e}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate ensemble on clean + adv data")
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

    defense_kwargs = dict(defense=args.defense, sigma=args.sigma)
    clip_values = load_clip_values()

    if args.at == "both":
        baseline = {"ensemble": evaluate_ensemble(
            at=False, device=args.device,
            clip_values=clip_values, **defense_kwargs)}
        at = {"ensemble": evaluate_ensemble(
            at=True, device=args.device,
            clip_values=clip_values, **defense_kwargs)}
        print_comparison_all(baseline, at)
        return

    use_at = args.at == "true"
    results = {"ensemble": evaluate_ensemble(
        at=use_at, device=args.device,
        clip_values=clip_values, **defense_kwargs)}
    print_summary(results)


if __name__ == "__main__":
    main()
