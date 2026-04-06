"""Train simple DNN surrogate → generate BB attacks (zoo, hsja) → evaluate
transfer to FOAMI ensemble.

Scenario: attacker has NO access to target models, trains a simple DNN on
the same dataset, runs BB attacks on it, and hopes they transfer.

Flow:
  1. Train SimpleDNN surrogate on train_tvae
  2. Generate zoo/hsja adversarial from surrogate (on test set)
  3. Evaluate transferability to FOAMI ensemble (baseline + AT)

Usage:
    python defense/surrogate_bb/evaluate_surrogate_bb.py --device mps
    python defense/surrogate_bb/evaluate_surrogate_bb.py --device mps --skip-train
    python defense/surrogate_bb/evaluate_surrogate_bb.py --device mps --attacks zoo
"""

import argparse
import os
import sys
import time

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
    BB_ATTACKS, ATTACK_GENERATORS,
)
from utils.evaluation import report_metrics, compute_asr
from utils.loaders import load_dataset
from utils.paths import load_attack_config, adv_eval_dir, model_path
from utils.masking import get_mutate_indices
from utils.ensemble import weighted_soft_voting, DEFAULT_WEIGHTS
from model import MODEL_REGISTRY
from model.surrogate_dnn import SurrogateDNNModel

logger = get_logger(__name__)

SURROGATE_NAME = "surrogate_dnn"
SURROGATE_FILE = f"framework_{SURROGATE_NAME}.pth"
_OUT_DIR = os.path.join(_IEC, "adv_samples", "adv_eval", SURROGATE_NAME)


# ── Attack generation ────────────────────────────────────────────────────────


def _get_generator(attack_name, classifier, config):
    import art_generator as ag
    cls_name = ATTACK_GENERATORS[attack_name]
    cls = getattr(ag, cls_name)
    return cls(classifier, generator_params=config)


def generate_bb_attacks(surrogate, attacks, X_test, y_test, feature_names,
                        mutate_indices, device):
    """Generate BB adversarial from surrogate, save as CSV."""
    scaler = surrogate.scaler
    X_sc = scaler.transform(X_test).astype(np.float32)

    art_clf = surrogate.wrap_for_art(X_sc, raw=False, device=device)

    # Surrogate clean accuracy
    art_clf_raw = surrogate.wrap_for_art(X_test, raw=True, device=device)
    preds_clean = np.argmax(art_clf_raw.predict(X_test), axis=1)
    from sklearn.metrics import accuracy_score, f1_score
    acc = accuracy_score(y_test, preds_clean)
    f1 = f1_score(y_test, preds_clean, average="macro")
    logger.info(f"Surrogate DNN clean acc={acc*100:.2f}%, F1={f1*100:.2f}%")

    os.makedirs(_OUT_DIR, exist_ok=True)
    metadata = {"feature_names": feature_names, "label_column": "Label"}

    generated_paths = {}
    for atk_name in attacks:
        logger.info(f"{'='*50}")
        logger.info(f"Generating {atk_name.upper()} from surrogate DNN")
        logger.info(f"{'='*50}")

        atk_cfg = load_attack_config(atk_name)
        gen = _get_generator(atk_name, art_clf, atk_cfg)

        start = time.time()
        df_adv_sc = gen.generate(X_sc, y_test, metadata, mutate_indices)
        elapsed = time.time() - start
        logger.info(f"{atk_name.upper()} runtime: {elapsed:.1f}s")

        # Inverse transform to raw space
        X_adv_sc = df_adv_sc[feature_names].values.astype(np.float32)
        X_adv_raw = scaler.inverse_transform(X_adv_sc).astype(np.float32)

        df_adv = pd.DataFrame(X_adv_raw, columns=feature_names)
        df_adv["Label"] = y_test

        filename = f"{SURROGATE_NAME}_{atk_name}_adv.csv"
        out_path = os.path.join(_OUT_DIR, filename)
        df_adv.to_csv(out_path, index=False)
        logger.info(f"Saved → {out_path}")
        generated_paths[atk_name] = out_path

        # ASR on surrogate itself
        preds_adv = np.argmax(art_clf_raw.predict(X_adv_raw), axis=1)
        correct = np.where(y_test == preds_clean)[0]
        if len(correct) > 0:
            asr = np.sum(preds_clean[correct] != preds_adv[correct]) / len(correct) * 100
            logger.info(f"{atk_name.upper()} → surrogate ASR={asr:.2f}%")

    return generated_paths


# ── FOAMI ensemble evaluation ────────────────────────────────────────────────


def _load_all_models(at=False, device="cpu"):
    models = {}
    for target in ALL_TARGETS:
        try:
            suffix = "_at" if at else ""
            ext = ".pkl" if target in TREE_TARGETS else ".pth"
            path = model_path(f"framework_{target}_TVAE{suffix}{ext}", at=at)
            models[target] = MODEL_REGISTRY[target].load(path, device=device)
            logger.info(f"Loaded {target} {'(AT)' if at else '(baseline)'}")
        except Exception as e:
            logger.error(f"Failed loading {target}: {e}")
    return models


def evaluate_ensemble_on_surrogate_adv(models, X_test, y_test, feature_names,
                                        adv_paths, at=False, device="cpu"):
    """Evaluate FOAMI ensemble on surrogate-generated adversarial samples.

    All models in ensemble see the SAME adversarial CSV (from surrogate).
    This is the transfer attack scenario.
    """
    variant = "Ensemble+AT" if at else "Ensemble"
    logger.info(f"{'='*60}")
    logger.info(f"Evaluating transfer: surrogate_dnn → {variant}")
    logger.info(f"{'='*60}")

    # Clean ensemble
    clean_probas = {}
    for name, m in models.items():
        art_clf = m.wrap_for_art(X_test, device=device)
        proba = m.art_predict_proba(art_clf, X_test)
        clean_probas[name] = proba

    _, y_ens_clean = weighted_soft_voting(clean_probas)
    clean_metrics = report_metrics(f"{variant} clean", y_test, y_ens_clean)
    results = {"clean": clean_metrics}

    # Adversarial transfer evaluation
    for atk_name, csv_path in adv_paths.items():
        try:
            df_adv = pd.read_csv(csv_path)
            X_adv = df_adv[feature_names].values.astype(np.float32)

            adv_probas = {}
            for name, m in models.items():
                art_clf = m.wrap_for_art(X_adv, device=device)
                proba = m.art_predict_proba(art_clf, X_adv)
                adv_probas[name] = proba

            _, y_ens_adv = weighted_soft_voting(adv_probas)
            metrics = report_metrics(
                f"{variant} surrogate_{atk_name}", y_test, y_ens_adv)
            asr = compute_asr(y_test, y_ens_clean, y_ens_adv)
            metrics["asr"] = asr
            logger.info(f"{'':>25s}  ASR={asr:6.2f}%")
            results[atk_name] = metrics

        except Exception as e:
            logger.error(f"Failed evaluating {atk_name}: {e}")

    return results


def evaluate_per_model(models, X_test, y_test, feature_names, adv_paths,
                       at=False, device="cpu"):
    """Evaluate each FOAMI model individually on surrogate adversarial."""
    tag = "AT" if at else "baseline"
    per_model = {}

    for target, m in models.items():
        art_clf_clean = m.wrap_for_art(X_test, device=device)
        proba_clean = m.art_predict_proba(art_clf_clean, X_test)
        y_clean = proba_clean.argmax(axis=1)

        model_results = {}
        for atk_name, csv_path in adv_paths.items():
            df_adv = pd.read_csv(csv_path)
            X_adv = df_adv[feature_names].values.astype(np.float32)

            art_clf = m.wrap_for_art(X_adv, device=device)
            proba_adv = m.art_predict_proba(art_clf, X_adv)
            y_adv = proba_adv.argmax(axis=1)

            asr = compute_asr(y_test, y_clean, y_adv)
            from sklearn.metrics import accuracy_score
            acc = accuracy_score(y_test, y_adv) * 100
            model_results[atk_name] = {"asr": asr, "acc": acc}
            logger.info(
                f"  {target}({tag}) ← surrogate_{atk_name}: "
                f"ASR={asr:.2f}%, Acc={acc:.2f}%"
            )

        per_model[target] = model_results

    return per_model


# ── Rich Tables ──────────────────────────────────────────────────────────────


def _print_ensemble_table(ens_base, ens_at, attacks):
    from rich.table import Table
    from rich.console import Console

    t = Table(
        title="Surrogate DNN BB Attack → FOAMI Ensemble Transfer",
        show_lines=True, title_style="bold cyan",
    )
    t.add_column("Metric", style="bold", min_width=24)
    t.add_column("Ensemble", justify="right", min_width=12)
    t.add_column("Ensemble+AT", justify="right", min_width=12)

    def _fmt(val):
        return f"{val:.2f}%" if val is not None else "-"

    for key, label in [("acc", "Acc (clean)"), ("f1", "F1 (clean)")]:
        row = [label]
        for rd in [ens_base, ens_at]:
            row.append(_fmt(rd.get("clean", {}).get(key)))
        t.add_row(*row)

    t.add_section()

    for atk in attacks:
        row = [f"ASR (surrogate_{atk})"]
        for rd in [ens_base, ens_at]:
            row.append(_fmt(rd.get(atk, {}).get("asr")))
        t.add_row(*row)

    t.add_section()

    for atk in attacks:
        row = [f"Acc (surrogate_{atk})"]
        for rd in [ens_base, ens_at]:
            row.append(_fmt(rd.get(atk, {}).get("acc")))
        t.add_row(*row)

    Console().print(t)
    Console().print()


def _print_per_model_table(per_model_base, per_model_at, attacks):
    from rich.table import Table
    from rich.console import Console

    for atk in attacks:
        t = Table(
            title=f"Per-Model Transfer: surrogate_dnn_{atk} → each FOAMI model",
            show_lines=True, title_style="bold cyan",
        )
        t.add_column("Model", style="bold", min_width=10)
        t.add_column("ASR (base)", justify="right", min_width=12)
        t.add_column("ASR (AT)", justify="right", min_width=12)
        t.add_column("Acc (base)", justify="right", min_width=12)
        t.add_column("Acc (AT)", justify="right", min_width=12)

        def _fmt(val):
            return f"{val:.2f}%" if val is not None else "-"

        for target in ALL_TARGETS:
            row = [target.upper()]
            for pm in [per_model_base, per_model_at]:
                row.append(_fmt(pm.get(target, {}).get(atk, {}).get("asr")))
            for pm in [per_model_base, per_model_at]:
                row.append(_fmt(pm.get(target, {}).get(atk, {}).get("acc")))
            t.add_row(*row)

        Console().print(t)
        Console().print()


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Surrogate DNN BB attack → FOAMI ensemble transfer eval")
    parser.add_argument("--attacks", "-a", nargs="+", default=BB_ATTACKS,
                        choices=BB_ATTACKS + ["all"],
                        help="BB attacks to generate (default: zoo hsja)")
    parser.add_argument("--device", "-d", default="auto",
                        choices=["cpu", "cuda", "mps", "auto"])
    parser.add_argument("--skip-train", action="store_true",
                        help="Load existing surrogate instead of retraining")
    parser.add_argument("--skip-gen", action="store_true",
                        help="Reuse existing adversarial CSVs")
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

    attacks = BB_ATTACKS if "all" in args.attacks else args.attacks

    # Load data
    _, X_train, y_train, _ = load_dataset("train_tvae")
    df_test, X_test, y_test, feature_names = load_dataset("test")
    # Always compute mutate_indices from TRAIN for consistency
    df_train_ref, *_ = load_dataset("train")
    mutate_indices = get_mutate_indices(df_train_ref)
    logger.info(f"Train: {X_train.shape}, Test: {X_test.shape}")

    # --- Phase 1: Train/load surrogate + generate BB adv ---
    save_path = model_path(SURROGATE_FILE)
    adv_paths = {}

    if args.skip_gen:
        # Reuse existing CSVs
        for atk in attacks:
            p = os.path.join(_OUT_DIR, f"{SURROGATE_NAME}_{atk}_adv.csv")
            if os.path.isfile(p):
                adv_paths[atk] = p
                logger.info(f"Reusing {p}")
            else:
                logger.error(f"Missing {p}, cannot skip generation")
                sys.exit(1)
    else:
        if args.skip_train and os.path.exists(save_path):
            logger.info(f"Loading existing surrogate from {save_path}")
            surrogate = SurrogateDNNModel.load(save_path, device=args.device)
        else:
            logger.info("Training surrogate DNN ...")
            surrogate = SurrogateDNNModel()
            surrogate.train(X_train, y_train, device=args.device)
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            surrogate.save(save_path)

        adv_paths = generate_bb_attacks(
            surrogate, attacks, X_test, y_test, feature_names,
            mutate_indices, args.device,
        )

    # --- Phase 2: Evaluate transfer to FOAMI ensemble ---
    models_base = _load_all_models(at=False, device=args.device)
    models_at = _load_all_models(at=True, device=args.device)

    ens_base = evaluate_ensemble_on_surrogate_adv(
        models_base, X_test, y_test, feature_names, adv_paths,
        device=args.device,
    )

    ens_at = evaluate_ensemble_on_surrogate_adv(
        models_at, X_test, y_test, feature_names, adv_paths,
        at=True, device=args.device,
    )

    # Per-model breakdown
    logger.info(f"\n{'#'*60}")
    logger.info("Per-model transfer breakdown:")
    logger.info(f"{'#'*60}")
    per_model_base = evaluate_per_model(
        models_base, X_test, y_test, feature_names, adv_paths,
        device=args.device,
    )
    per_model_at = evaluate_per_model(
        models_at, X_test, y_test, feature_names, adv_paths,
        at=True, device=args.device,
    )

    # Print results
    _print_ensemble_table(ens_base, ens_at, attacks)
    _print_per_model_table(per_model_base, per_model_at, attacks)

    logger.info("Done!")


if __name__ == "__main__":
    main()
