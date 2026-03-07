"""Evaluate ensemble and MI models with per-component adversarial inputs.

Each component model receives the adversarial X generated specifically against
it (or a fallback DL model), predicts independently, then probabilities are
combined — matching the pattern in apelid/classifier_parallel_ens_eval.py.

Pattern:
    attack=pgd
        cat    → adv_training/cat/cat_pgd_adv.csv   (missing → fallback lstm)
        rf     → adv_training/rf/rf_pgd_adv.csv     (missing → fallback lstm)
        lstm   → adv_training/lstm/lstm_pgd_adv.csv ✓
        resdnn → adv_training/resdnn/resdnn_pgd_adv.csv ✓

        ensemble: Σ weight_i × model_i.predict_proba(adv_X_i)
        MI:       MI_mechanism(cat(adv_X_cat), rf(adv_X_rf),
                               lstm(adv_X_lstm), resdnn(adv_X_resdnn))

Fallback priority for missing adv CSV: lstm → resdnn → skip model

Usage:
    python evaluate_ensemble_mi.py --attack pgd fgsm zoo hsja

    # After adversarial training (AT models staged in models_at/)
    python evaluate_ensemble_mi.py --attack pgd zoo --models-dir ../../models_at

    # Save results
    python evaluate_ensemble_mi.py --attack pgd zoo --output-csv results.csv
"""

import os
import sys
import argparse

# ── macOS / PyTorch compatibility (must be set before torch is imported) ────────
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from prettytable import PrettyTable

# ── Path bootstrap ──────────────────────────────────────────────────────────────
_HERE  = os.path.dirname(os.path.realpath(__file__))
_FOAMI = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, _FOAMI)

from utils.paths import setup_paths, MODELS_DIR, ADV_EVAL_DIR, TEST_CSV
setup_paths()

from utils.logging   import setup_logging, get_logger
from utils.loaders   import load_wrapper, load_features_csv, resolve_adv_path, parse_ensemble_config
from utils.evaluation import asr
from utils.ensemble  import (
    ENSEMBLE_COMPONENTS, MI_GBT, MI_DL,
    weighted_combine, mi_combine,
)
from utils.constants import ALL_ATTACKS

logger = get_logger(__name__)


# ── Load component wrappers ─────────────────────────────────────────────────────

def _load_components(models_dir: str, clip_values: tuple,
                     num_classes: int, input_dim: int,
                     device: str, components: list) -> dict:
    wrappers = {}
    for t in components:
        try:
            wrappers[t] = load_wrapper(t, models_dir, clip_values,
                                       num_classes, input_dim, device)
            logger.info(f"  Loaded: {t}")
        except SystemExit as e:
            logger.warning(f"  [!] Cannot load {t}: {e}")
    return wrappers


# ── Per-attack evaluation ───────────────────────────────────────────────────────

def evaluate_attack(
    attack: str,
    wrappers: dict,
    plain_preds: np.ndarray,
    y_plain: np.ndarray,
    adv_dir: str,
    num_classes: int,
    ew: dict,
    mi_cfg: dict,
    w_gbt_base: np.ndarray,
    max_workers: int,
    mode: str,   # 'ensemble' or 'mi'
) -> 'dict | None':

    # ── 1. Resolve per-component adv CSV ─────────────────────────────────────
    components = list(wrappers.keys())
    adv_paths  = {}
    for c in components:
        p = resolve_adv_path(c, attack, adv_dir)
        if p is None:
            logger.warning(f"  [{attack}] No adv CSV for {c} — skip attack")
        adv_paths[c] = p

    if any(v is None for v in adv_paths.values()):
        logger.warning(f"  [{attack}] Skipping: missing adv CSV for some components")
        return None

    # ── 2. Load per-component adv data in parallel ────────────────────────────
    def _load(args):
        c, path = args
        X, y = load_features_csv(path)
        return c, X, y

    adv_data: dict = {}
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(_load, (c, p)): c for c, p in adv_paths.items()}
        for fut in as_completed(futs):
            c = futs[fut]
            try:
                c_out, X, y = fut.result()
                adv_data[c_out] = (X, y)
                logger.debug(f"  [{attack}] Loaded {c_out}: {X.shape}")
            except Exception as e:
                logger.warning(f"  [{attack}] Load failed for {c}: {e}")

    if set(adv_data.keys()) != set(components):
        logger.warning(f"  [{attack}] Skipping: some adv loads failed")
        return None

    # ── 3. Predict per component on its own adv X in parallel ─────────────────
    def _predict(args):
        c, wrapper = args
        X_adv = adv_data[c][0]
        return c, wrapper.predict_proba(X_adv)

    proba_map: dict = {}
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(_predict, (c, w)): c for c, w in wrappers.items()}
        for fut in as_completed(futs):
            c = futs[fut]
            try:
                c_out, proba = fut.result()
                proba_map[c_out] = proba
            except Exception as e:
                logger.warning(f"  [{attack}] Predict failed for {c}: {e}")

    if set(proba_map.keys()) != set(components):
        logger.warning(f"  [{attack}] Skipping: some predictions failed")
        return None

    # ── 4. Combine probabilities ───────────────────────────────────────────────
    if mode == 'ensemble':
        _, adv_preds = weighted_combine(proba_map, ew, num_classes)
    else:  # mi
        p_ens = mi_combine(proba_map, num_classes, w_gbt_base,
                           mi_cfg['alpha'], mi_cfg['beta'], mi_cfg['threshold'])
        adv_preds = p_ens.argmax(axis=1).astype(np.int64)

    # ── 5. Metrics ────────────────────────────────────────────────────────────
    y_adv   = adv_data[next(iter(adv_data))][1]   # labels (same src data for all)
    adv_acc = float(accuracy_score(y_adv, adv_preds))
    adv_f1  = float(f1_score(y_adv, adv_preds, average='macro', zero_division=0))
    asr_val = asr(plain_preds, adv_preds, y_plain)

    logger.info(f"  [{attack}] Adv Acc={adv_acc*100:.2f}%  "
                f"Macro-F1={adv_f1*100:.2f}%  ASR={asr_val*100:.2f}%")

    return {
        'attack':  attack,
        'adv_acc': adv_acc,
        'adv_f1':  adv_f1,
        'asr':     asr_val,
    }


# ── Main ─────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate ensemble / MI with per-component adversarial inputs"
    )
    parser.add_argument('--attack', '-a', nargs='+', required=True,
                        choices=list(ALL_ATTACKS),
                        help="Attack(s) to evaluate")
    parser.add_argument('--models-dir', default=None,
                        help=f"Model directory (default: {MODELS_DIR})")
    parser.add_argument('--adv-dir', default=None,
                        help=f"Adv samples root (default: {ADV_EVAL_DIR})")
    parser.add_argument('--plain-in', default=None,
                        help=f"Plain test CSV (default: {TEST_CSV})")
    parser.add_argument('--device', default='cpu', choices=['cpu', 'cuda', 'auto'])
    parser.add_argument('--max-workers', type=int, default=4)
    parser.add_argument('--ensemble-weights', type=str, default=None,
                        help="JSON dict of ensemble weights")
    parser.add_argument('--mi-params', type=str, default=None,
                        help="JSON dict: alpha, beta, threshold, w_gbt_base")
    parser.add_argument('--targets', nargs='+', default=['ensemble', 'mi'],
                        choices=['ensemble', 'mi'],
                        help="Which targets to evaluate (default: both)")
    parser.add_argument('--output-csv', default=None)
    parser.add_argument('--log-level', default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'])
    args = parser.parse_args()

    setup_logging(args.log_level)

    models_dir = args.models_dir or MODELS_DIR
    adv_dir    = args.adv_dir    or ADV_EVAL_DIR
    plain_in   = args.plain_in   or TEST_CSV

    # ── Parse configs ─────────────────────────────────────────────────────────
    ew, mi_cfg, w_gbt_base = parse_ensemble_config(args)

    # ── Load plain test data ──────────────────────────────────────────────────
    logger.info(f"[+] Plain test: {plain_in}")
    X_plain, y_plain = load_features_csv(plain_in)
    num_classes = len(np.unique(y_plain))
    input_dim   = X_plain.shape[1]
    clip_values = (float(X_plain.min()), float(X_plain.max()))
    logger.info(f"[+] Shape={X_plain.shape}, classes={num_classes}")

    # ── Component wrappers for ensemble and MI ────────────────────────────────
    logger.info("[+] Loading component models ...")
    all_components = sorted(set(ENSEMBLE_COMPONENTS + MI_GBT + MI_DL))
    wrappers = _load_components(models_dir, clip_values, num_classes,
                                input_dim, args.device, all_components)

    ens_wrappers = {k: wrappers[k] for k in ENSEMBLE_COMPONENTS if k in wrappers}
    mi_gbt       = {k: wrappers[k] for k in MI_GBT if k in wrappers}
    mi_dl        = {k: wrappers[k] for k in MI_DL  if k in wrappers}
    mi_wrappers  = {**mi_gbt, **mi_dl}

    all_results = {}

    for mode in args.targets:
        logger.info(f"\n{'='*60}")
        logger.info(f"  TARGET: {mode.upper()}")
        logger.info(f"{'='*60}")

        if mode == 'ensemble':
            active_wrappers = ens_wrappers
            active_ew       = {k: ew.get(k, 0.0) for k in ens_wrappers}
        else:
            active_wrappers = mi_wrappers
            active_ew       = {}  # MI uses its own mechanism

        if not active_wrappers:
            logger.warning(f"  [!] No component models loaded for {mode} — skip")
            continue

        # ── Plain baseline ────────────────────────────────────────────────────
        logger.info("[+] Plain baseline ...")
        plain_proba_map: dict = {}
        for c, w in active_wrappers.items():
            plain_proba_map[c] = w.predict_proba(X_plain)

        if mode == 'ensemble':
            _, plain_preds = weighted_combine(plain_proba_map, active_ew, num_classes)
        else:
            p_ens = mi_combine(plain_proba_map, num_classes, w_gbt_base,
                               mi_cfg['alpha'], mi_cfg['beta'], mi_cfg['threshold'])
            plain_preds = p_ens.argmax(axis=1).astype(np.int64)

        plain_acc = float(accuracy_score(y_plain, plain_preds))
        plain_f1  = float(f1_score(y_plain, plain_preds, average='macro', zero_division=0))
        logger.info(f"  Plain Acc={plain_acc*100:.2f}%  Macro-F1={plain_f1*100:.2f}%")

        # ── Per-attack evaluation ─────────────────────────────────────────────
        results = []
        for atk in args.attack:
            logger.info(f"\n[+] Attack: {atk}")
            res = evaluate_attack(
                attack=atk,
                wrappers=active_wrappers,
                plain_preds=plain_preds,
                y_plain=y_plain,
                adv_dir=adv_dir,
                num_classes=num_classes,
                ew=active_ew,
                mi_cfg=mi_cfg,
                w_gbt_base=w_gbt_base,
                max_workers=args.max_workers,
                mode=mode,
            )
            if res is not None:
                res['plain_acc'] = plain_acc
                res['plain_f1']  = plain_f1
                results.append(res)

        all_results[mode] = results

        # ── Table ─────────────────────────────────────────────────────────────
        if results:
            tbl = PrettyTable(['Attack', 'Plain Acc', 'Plain F1',
                               'Adv Acc', 'Adv F1', 'ASR'])
            tbl.align = 'r'
            tbl.align['Attack'] = 'l'
            tbl.add_row(['original',
                         f"{plain_acc*100:.2f}%", f"{plain_f1*100:.2f}%",
                         '—', '—', '—'])
            for r in results:
                tbl.add_row([r['attack'],
                             f"{r['plain_acc']*100:.2f}%", f"{r['plain_f1']*100:.2f}%",
                             f"{r['adv_acc']*100:.2f}%",   f"{r['adv_f1']*100:.2f}%",
                             f"{r['asr']*100:.2f}%"])
            logger.info(f"\n[{mode.upper()}]\n" + tbl.get_string())

    # ── Save CSV ──────────────────────────────────────────────────────────────
    if args.output_csv:
        rows = [{'target': mode, **r}
                for mode, rs in all_results.items() for r in rs]
        if rows:
            pd.DataFrame(rows).to_csv(args.output_csv, index=False)
            logger.info(f"[+] Saved → {args.output_csv}")


if __name__ == '__main__':
    main()
