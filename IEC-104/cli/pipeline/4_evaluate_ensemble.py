"""Evaluate Weighted Soft Voting ensemble on clean + adversarial data.

For each attack, every model predicts on its own adversarial CSV
(following the same mapping as 2_evaluate.py):
  - Tree (cat, rf): self BB (zoo, hsja) + transfer WB from resdnn_sc
  - DL (lstm, resdnn): self WB+BB from {target}_sc

Then weighted soft voting combines the 4 probas into ensemble prediction.
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
from utils.defense import create_preprocessing_defense
from model import MODEL_REGISTRY

logger = get_logger(__name__)


# ── Model Loading ────────────────────────────────────────────────────────────


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


# ── Per-model adversarial CSV mapping ────────────────────────────────────────


def _adv_csv_path(target, attack):
    """Resolve the correct adversarial CSV path for a (target, attack) pair.

    Same logic as 2_evaluate.py _find_adv_csvs:
      - DL: adv_eval/{target}_sc/{target}_{attack}_adv.csv
      - Tree self BB: adv_eval/{target}/{target}_{attack}_adv.csv
      - Tree transfer WB: adv_eval/resdnn_sc/resdnn_{attack}_adv.csv
    """
    if target in DL_TARGETS:
        return adv_eval_dir(f"{target}_sc", f"{target}_{attack}_adv.csv")
    else:
        # Tree: BB → self, WB → transfer from resdnn_sc
        if attack in BB_ATTACKS:
            return adv_eval_dir(target, f"{target}_{attack}_adv.csv")
        else:
            return adv_eval_dir("resdnn_sc", f"resdnn_{attack}_adv.csv")


def _discover_attacks():
    """Discover all attacks that have adversarial CSVs for every model.

    Returns list of (attack_name, label) where label is used for display.
    Only includes attacks where ALL 4 models have a valid CSV.
    """
    attacks = []

    # Collect all candidate attacks from DL targets (they support all attacks)
    candidate_attacks = set()
    for dl_target in DL_TARGETS:
        d = adv_eval_dir(f"{dl_target}_sc")
        if not os.path.isdir(d):
            continue
        for fname in os.listdir(d):
            if not fname.endswith("_adv.csv") or "surrogate" in fname:
                continue
            # e.g. resdnn_pgd_adv.csv → pgd
            basename = fname.replace("_adv.csv", "")
            prefix = dl_target + "_"
            atk = basename[len(prefix):] if basename.startswith(prefix) else basename
            candidate_attacks.add(atk)

    # Add BB attacks from tree targets
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

    # Filter: only attacks where ALL models have a valid CSV
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


# ── Ensemble Evaluation ──────────────────────────────────────────────────────


def evaluate_ensemble(at=False, device="cpu", defense=None, sigma=0.01):
    tag = "ENSEMBLE (AT)" if at else "ENSEMBLE (baseline)"
    if defense:
        tag += f" +{defense}(σ={sigma})"
    logger.info(f"{'='*60}")
    logger.info(f"Evaluating: {tag}")
    logger.info(f"Weights: {DEFAULT_WEIGHTS}")
    logger.info(f"{'='*60}")

    _, X_test, y_test, feature_names = load_dataset("test")

    models = _load_all_models(at=at, device=device)
    if len(models) < len(ALL_TARGETS):
        logger.error(f"Need all {len(ALL_TARGETS)} models for ensemble, "
                     f"got {len(models)}")
        return {}

    # Always use ART classifiers; defense is applied when specified
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

    _, y_ens_clean = weighted_soft_voting(clean_probas)
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
            # Each model predicts on its own adversarial CSV
            adv_probas = {}
            for target in models:
                csv_path = _adv_csv_path(target, atk)
                df_adv = pd.read_csv(csv_path)
                X_adv = df_adv[feature_names].values.astype(np.float32)
                adv_probas[target] = proba_fn(target, X_adv)
                logger.debug(f"  {target} → {os.path.basename(csv_path)}")

            _, y_ens_adv = weighted_soft_voting(adv_probas)

            # Use y_test as ground truth (adv samples derived from test set)
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
    parser = argparse.ArgumentParser(description="Evaluate ensemble on clean + adv data")
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

    defense_kwargs = dict(defense=args.defense, sigma=args.sigma)

    if args.at == "both":
        baseline = {"ensemble": evaluate_ensemble(at=False, device=args.device, **defense_kwargs)}
        at = {"ensemble": evaluate_ensemble(at=True, device=args.device, **defense_kwargs)}
        print_comparison_all(baseline, at)
        return {"baseline": baseline, "at": at}

    use_at = args.at == "true"
    results = {"ensemble": evaluate_ensemble(at=use_at, device=args.device, **defense_kwargs)}
    print_summary(results)
    return results


if __name__ == "__main__":
    main()
