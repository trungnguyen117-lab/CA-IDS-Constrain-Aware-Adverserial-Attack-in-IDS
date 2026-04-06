"""Evaluate KMeans Probability Correction + Ensemble defense.

V4: Instead of adjusting weights, directly correct each model's probability
vector before ensemble voting. KMeans cluster class distributions serve as
"second opinion" — when model disagrees with cluster, blend proba toward
cluster distribution. This changes WHAT models predict, not how much
we trust them.

4-way: Ensemble | Ensemble+AT | KMeans-Corrected | KMeans-Corrected+AT

Usage:
    python defense/kmeans_ensemble/evaluate_kmeans_ensemble.py --device mps --at both
    python defense/kmeans_ensemble/evaluate_kmeans_ensemble.py --device mps --at true \
        --sweep 0.0 0.25 0.5 0.75 1.0
    python defense/kmeans_ensemble/evaluate_kmeans_ensemble.py --device mps --at true \
        --sweep-k 12 36 96 --sweep 0.5 0.75
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import yaml

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
from utils.kmeans_defense import KMeansDefense
from model import MODEL_REGISTRY

logger = get_logger(__name__)


# ── Config ────────────────────────────────────────────────────────────────────


def load_config():
    cfg_path = os.path.join(_HERE, "config.yaml")
    if os.path.isfile(cfg_path):
        with open(cfg_path) as f:
            return yaml.safe_load(f)
    return {}


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


# ── Adversarial CSV resolution ────────────────────────────────────────────────


def _adv_csv_path(target, attack):
    if target in DL_TARGETS:
        return adv_eval_dir(f"{target}_sc", f"{target}_{attack}_adv.csv")
    else:
        if attack in BB_ATTACKS:
            return adv_eval_dir(target, f"{target}_{attack}_adv.csv")
        else:
            return adv_eval_dir("resdnn_sc", f"resdnn_{attack}_adv.csv")


def _discover_attacks():
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
        all_exist = True
        for target in ALL_TARGETS:
            if not os.path.isfile(_adv_csv_path(target, atk)):
                all_exist = False
                break
        if all_exist:
            attacks.append(atk)

    return attacks


# ── Evaluation Core ──────────────────────────────────────────────────────────


def evaluate_variant(models, X_test, y_test, feature_names, attacks,
                     kmeans_def=None, alpha=0.0, at=False, device="cpu"):
    """Evaluate one variant: standard ensemble or KMeans-corrected ensemble.

    When kmeans_def is provided and alpha > 0:
      1. Each model predicts proba on its adversarial CSV
      2. correct_proba(X_adv, proba, alpha) blends proba toward cluster dist
      3. Standard weighted_soft_voting on corrected probas
    """
    use_correction = kmeans_def is not None and alpha > 0
    variant = "KMeans-Corrected" if use_correction else "Ensemble"
    if at:
        variant += "+AT"
    if use_correction:
        variant += f"(α={alpha})"

    tag = variant
    logger.info(f"{'='*60}")
    logger.info(f"Evaluating: {tag}")
    logger.info(f"{'='*60}")

    # --- Clean evaluation ---
    clean_probas = {}
    for name, m in models.items():
        art_clf = m.wrap_for_art(X_test, device=device)
        proba = m.art_predict_proba(art_clf, X_test)
        if use_correction:
            proba = kmeans_def.correct_proba(X_test, proba, alpha=alpha)
        clean_probas[name] = proba

    _, y_ens_clean = weighted_soft_voting(clean_probas)
    clean_metrics = report_metrics(f"{tag} clean", y_test, y_ens_clean)
    results = {"clean": clean_metrics}

    # --- Adversarial evaluation ---
    if not attacks:
        logger.info("No adversarial attacks found.")
        return results

    for atk in attacks:
        try:
            adv_probas = {}
            for target in models:
                csv_path = _adv_csv_path(target, atk)
                df_adv = pd.read_csv(csv_path)
                X_adv = df_adv[feature_names].values.astype(np.float32)

                art_clf = models[target].wrap_for_art(X_adv, device=device)
                proba = models[target].art_predict_proba(art_clf, X_adv)

                if use_correction:
                    proba_before = proba.copy()
                    proba = kmeans_def.correct_proba(X_adv, proba, alpha=alpha)

                    # Log how many predictions changed
                    pred_before = proba_before.argmax(axis=1)
                    pred_after = proba.argmax(axis=1)
                    n_changed = (pred_before != pred_after).sum()
                    logger.info(
                        f"  {target}/{atk}: {n_changed}/{len(X_adv)} "
                        f"predictions changed by correction"
                    )

                adv_probas[target] = proba

            _, y_ens_adv = weighted_soft_voting(adv_probas)

            metrics = report_metrics(f"{tag} {atk}", y_test, y_ens_adv)
            asr = compute_asr(y_test, y_ens_clean, y_ens_adv)
            metrics["asr"] = asr
            logger.info(f"{'':>25s}  ASR={asr:6.2f}%")
            results[atk] = metrics

        except Exception as e:
            logger.error(f"Failed evaluating {tag} on {atk}: {e}")

    return results


# ── Rich Tables ──────────────────────────────────────────────────────────────


def _print_four_way(ens_base, ens_at, km_base, km_at):
    from rich.table import Table
    from rich.console import Console

    all_attacks = []
    for rd in [ens_base, ens_at, km_base, km_at]:
        for name in rd:
            if name not in all_attacks:
                all_attacks.append(name)

    t = Table(
        title="KMeans Probability Correction + Ensemble (v4)",
        show_lines=True, title_style="bold cyan",
    )
    t.add_column("Metric", style="bold", min_width=18)
    t.add_column("Ensemble", justify="right", min_width=12)
    t.add_column("Ensemble+AT", justify="right", min_width=12)
    t.add_column("KM-Corrected", justify="right", min_width=12)
    t.add_column("KM-Corr+AT", justify="right", min_width=12)

    def _fmt(val):
        return f"{val:.2f}%" if val is not None else "-"

    for key, label in [("acc", "Acc (clean)"), ("f1", "F1 (clean)")]:
        row = [label]
        for rd in [ens_base, ens_at, km_base, km_at]:
            row.append(_fmt(rd.get("clean", {}).get(key)))
        t.add_row(*row)

    t.add_section()

    for atk in all_attacks:
        if atk == "clean":
            continue
        row = [f"ASR ({atk})"]
        for rd in [ens_base, ens_at, km_base, km_at]:
            row.append(_fmt(rd.get(atk, {}).get("asr")))
        t.add_row(*row)

    t.add_section()

    for atk in all_attacks:
        if atk == "clean":
            continue
        row = [f"Acc ({atk})"]
        for rd in [ens_base, ens_at, km_base, km_at]:
            row.append(_fmt(rd.get(atk, {}).get("acc")))
        t.add_row(*row)

    Console().print(t)
    Console().print()


def _print_sweep(sweep_results, attacks):
    from rich.table import Table
    from rich.console import Console

    labels = list(sweep_results.keys())
    t = Table(
        title="Parameter Sweep — KMeans Probability Correction (v4)",
        show_lines=True, title_style="bold cyan",
    )
    t.add_column("Metric", style="bold", min_width=18)
    for label in labels:
        t.add_column(label, justify="right", min_width=10)

    def _fmt(val):
        return f"{val:.2f}%" if val is not None else "-"

    for key, metric in [("acc", "Acc (clean)"), ("f1", "F1 (clean)")]:
        row = [metric]
        for label in labels:
            row.append(_fmt(sweep_results[label].get("clean", {}).get(key)))
        t.add_row(*row)

    t.add_section()

    for atk in attacks:
        row = [f"ASR ({atk})"]
        for label in labels:
            row.append(_fmt(sweep_results[label].get(atk, {}).get("asr")))
        t.add_row(*row)

    t.add_section()

    for atk in attacks:
        row = [f"Acc ({atk})"]
        for label in labels:
            row.append(_fmt(sweep_results[label].get(atk, {}).get("acc")))
        t.add_row(*row)

    Console().print(t)
    Console().print()


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate KMeans Probability Correction + Ensemble (v4)")
    parser.add_argument("--at", default="both",
                        choices=["false", "true", "both"])
    parser.add_argument("--device", "-d", default="auto",
                        choices=["cpu", "cuda", "mps", "auto"])
    parser.add_argument("--k", type=int, default=None,
                        help="Number of KMeans clusters")
    parser.add_argument("--alpha", type=float, default=None,
                        help="Correction strength (0=none, 1=full)")
    parser.add_argument("--threshold-pct", type=float, default=None)
    parser.add_argument("--sweep", nargs="+", type=float, default=None,
                        help="Sweep alpha values")
    parser.add_argument("--sweep-k", nargs="+", type=int, default=None,
                        help="Sweep k values")
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

    cfg = load_config()
    n_clusters = args.k or cfg.get("n_clusters", 36)
    alpha = args.alpha if args.alpha is not None else cfg.get("alpha", 0.5)
    threshold_pct = args.threshold_pct or cfg.get("threshold_pct", 50)

    logger.info(f"Config: k={n_clusters}, alpha={alpha}, "
                f"threshold_pct={threshold_pct}")

    # Load data
    _, X_train, y_train, _ = load_dataset("train_tvae")
    _, X_test, y_test, feature_names = load_dataset("test")

    # Fit KMeans
    kmeans_def = KMeansDefense(
        n_clusters=n_clusters, threshold_pct=threshold_pct)
    kmeans_def.fit(X_train, y_train)

    save_path = os.path.join(_HERE, "kmeans_defense.pkl")
    kmeans_def.save(save_path)

    attacks = _discover_attacks()
    if not attacks:
        logger.error("No adversarial attacks found.")
        sys.exit(1)
    logger.info(f"Found {len(attacks)} attacks: {attacks}")

    # --- Sweep mode ---
    has_sweep = args.sweep or args.sweep_k
    if has_sweep:
        use_at = args.at in ("true", "both")
        models = _load_all_models(at=use_at, device=args.device)
        if len(models) < len(ALL_TARGETS):
            logger.error("Cannot load all models.")
            sys.exit(1)

        alpha_vals = args.sweep or [alpha]
        k_vals = args.sweep_k or [n_clusters]

        sweep_results = {}
        for k_val in k_vals:
            if k_val != n_clusters or len(k_vals) > 1:
                km = KMeansDefense(n_clusters=k_val, threshold_pct=threshold_pct)
                km.fit(X_train, y_train)
            else:
                km = kmeans_def

            for a in alpha_vals:
                label = f"k={k_val},α={a}"
                logger.info(f"\n--- {label} ---")
                sweep_results[label] = evaluate_variant(
                    models, X_test, y_test, feature_names, attacks,
                    kmeans_def=km, alpha=a,
                    at=use_at, device=args.device,
                )

        _print_sweep(sweep_results, attacks)
        return

    # --- Standard 4-way comparison ---
    models_base = _load_all_models(at=False, device=args.device)
    if len(models_base) < len(ALL_TARGETS):
        logger.error("Cannot load all baseline models.")
        sys.exit(1)

    ens_base = evaluate_variant(
        models_base, X_test, y_test, feature_names, attacks,
        device=args.device,
    )

    km_base = evaluate_variant(
        models_base, X_test, y_test, feature_names, attacks,
        kmeans_def=kmeans_def, alpha=alpha,
        device=args.device,
    )

    ens_at = {}
    km_at = {}

    if args.at in ("true", "both"):
        models_at = _load_all_models(at=True, device=args.device)
        if len(models_at) >= len(ALL_TARGETS):
            ens_at = evaluate_variant(
                models_at, X_test, y_test, feature_names, attacks,
                at=True, device=args.device,
            )
            km_at = evaluate_variant(
                models_at, X_test, y_test, feature_names, attacks,
                kmeans_def=kmeans_def, alpha=alpha,
                at=True, device=args.device,
            )
        else:
            logger.warning("Could not load all AT models.")

    _print_four_way(ens_base, ens_at, km_base, km_at)


if __name__ == "__main__":
    main()
