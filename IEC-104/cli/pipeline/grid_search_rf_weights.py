"""Grid search over AT weights & clean_adv_ratio for RF.

Iterates combinations of:
  - AT_CLEAN_ADV_RATIO["rf"]
  - AT_WEIGHTS_ML (scale by attack group)
  - AT_BB_WEIGHT_ZOO, AT_BB_WEIGHT_HSJA

For each combo: assemble → retrain RF → evaluate → record metrics.
Outputs ranked CSV + console table.
"""

import argparse
import csv
import itertools
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_CLI = os.path.dirname(_HERE)
_IEC = os.path.dirname(_CLI)
sys.path.insert(0, _IEC)
sys.path.insert(0, os.path.join(_IEC, "script"))
sys.path.insert(0, _CLI)

from utils.logging import setup_logging, get_logger
import utils.constants as C
from utils.loaders import load_dataset
from utils.paths import set_version

logger = get_logger(__name__)

# ── Search Space ────────────────────────────────────────────────────────────

RATIOS = [0.08, 0.10, 0.12, 0.15, 0.20]

# Scales applied to each transfer-attack weight (multiplied to base weight)
TRANSFER_SCALES = {
    "resdnn_deepfool": [1.0, 1.2, 1.4],
    "resdnn_fgsm":     [1.0, 1.1, 1.2],
    "resdnn_pgd":      [1.0, 1.1],
    "resdnn_cw":       [0.8, 0.9, 1.0],
    "resdnn_mim":      [0.8, 0.9, 1.0],
    "lstm_pgd":        [1.0],
    "lstm_fgsm":       [0.8, 1.0],
}

BB_ZOO_VALS = [5.0, 5.5, 6.0]
BB_HSJA_VALS = [5.0, 6.0, 7.0]


def _build_search_space():
    """Build manageable search space with grouped scales to avoid combinatorial explosion.

    Groups attacks by remaining ASR level:
      high_asr: deepfool, fgsm  (8-9% remaining ASR)
      mid_asr:  pgd, cw, mim   (5-6%)
    """
    high_scales = [1.0, 1.2, 1.4]   # for deepfool, fgsm
    mid_scales = [0.8, 0.9, 1.0]    # for cw, mim
    configs = []
    for ratio in RATIOS:
        for h_scale in high_scales:
            for m_scale in mid_scales:
                for bb_zoo in BB_ZOO_VALS:
                    for bb_hsja in BB_HSJA_VALS:
                        weights = {}
                        base = C.AT_WEIGHTS_ML.copy()
                        for k, v in base.items():
                            if "deepfool" in k or "fgsm" in k:
                                weights[k] = round(v * h_scale, 1)
                            elif "cw" in k or "mim" in k:
                                weights[k] = round(v * m_scale, 1)
                            else:
                                weights[k] = v
                        configs.append({
                            "ratio": ratio,
                            "weights_ml": weights,
                            "bb_zoo": bb_zoo,
                            "bb_hsja": bb_hsja,
                            "h_scale": h_scale,
                            "m_scale": m_scale,
                        })
    return configs


def _patch_constants(cfg):
    """Monkey-patch constants in both the module and any importers."""
    import pipeline_3_adv_train as p3

    C.AT_CLEAN_ADV_RATIO["rf"] = cfg["ratio"]
    C.AT_WEIGHTS_ML.clear()
    C.AT_WEIGHTS_ML.update(cfg["weights_ml"])
    C.AT_BB_WEIGHT_ZOO = cfg["bb_zoo"]
    C.AT_BB_WEIGHT_HSJA = cfg["bb_hsja"]
    # Scalars are copied by value at import — patch the importer too
    p3.AT_BB_WEIGHT_ZOO = cfg["bb_zoo"]
    p3.AT_BB_WEIGHT_HSJA = cfg["bb_hsja"]


def _restore_constants(orig):
    """Restore original constant values."""
    import pipeline_3_adv_train as p3

    C.AT_CLEAN_ADV_RATIO["rf"] = orig["ratio"]
    C.AT_WEIGHTS_ML.clear()
    C.AT_WEIGHTS_ML.update(orig["weights_ml"])
    C.AT_BB_WEIGHT_ZOO = orig["bb_zoo"]
    C.AT_BB_WEIGHT_HSJA = orig["bb_hsja"]
    p3.AT_BB_WEIGHT_ZOO = orig["bb_zoo"]
    p3.AT_BB_WEIGHT_HSJA = orig["bb_hsja"]


def _run_one(cfg, device="cpu"):
    """Assemble AT data, retrain RF, evaluate. Returns metrics dict."""
    from pipeline_3_adv_train import assemble_ml_at_data, _assemble_clean_and_adv
    from pipeline_3_adv_train import retrain_model as _retrain
    from pipeline_2_evaluate import evaluate_target

    _patch_constants(cfg)

    # Assemble + retrain
    df_merged = assemble_ml_at_data("rf")
    _retrain("rf", df_merged, device=device)

    # Evaluate AT model
    results = evaluate_target("rf", at=True, device=device)

    clean_acc = results.get("clean", {}).get("acc", 0)
    clean_f1 = results.get("clean", {}).get("f1", 0)

    asrs = {}
    for k, v in results.items():
        if k != "clean" and "asr" in v:
            asrs[k] = v["asr"]

    mean_asr = np.mean(list(asrs.values())) if asrs else 100.0
    max_asr = max(asrs.values()) if asrs else 100.0

    return {
        "clean_acc": clean_acc,
        "clean_f1": clean_f1,
        "mean_asr": mean_asr,
        "max_asr": max_asr,
        "asrs": asrs,
    }


def main():
    parser = argparse.ArgumentParser(description="Grid search RF AT weights")
    parser.add_argument("--device", "-d", default="cpu")
    parser.add_argument("--version", "-V", default="v1")
    parser.add_argument("--output", "-o", default="grid_search_rf_results.csv")
    parser.add_argument("--top", type=int, default=10, help="Show top N results")
    parser.add_argument("--log-level", default="WARNING",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    setup_logging(args.log_level)
    set_version(args.version)

    # Save originals to restore after each trial
    orig = {
        "ratio": C.AT_CLEAN_ADV_RATIO["rf"],
        "weights_ml": C.AT_WEIGHTS_ML.copy(),
        "bb_zoo": C.AT_BB_WEIGHT_ZOO,
        "bb_hsja": C.AT_BB_WEIGHT_HSJA,
    }

    # Import pipeline modules after path setup
    # We need to import them as modules since they're scripts
    import importlib.util

    for mod_name, filename in [
        ("pipeline_3_adv_train", "3_adv_train.py"),
        ("pipeline_2_evaluate", "2_evaluate.py"),
    ]:
        spec = importlib.util.spec_from_file_location(
            mod_name, os.path.join(_HERE, filename))
        mod = importlib.util.module_from_spec(spec)
        sys.modules[mod_name] = mod
        spec.loader.exec_module(mod)

    configs = _build_search_space()
    logger.warning(f"Grid search: {len(configs)} configurations")

    rows = []
    best_score = -float("inf")

    for i, cfg in enumerate(configs):
        t0 = time.time()
        try:
            metrics = _run_one(cfg, device=args.device)
        except Exception as e:
            logger.error(f"[{i+1}/{len(configs)}] FAILED: {e}")
            _restore_constants(orig)
            continue
        finally:
            _restore_constants(orig)

        elapsed = time.time() - t0

        # Composite score: maximize clean_acc, minimize mean_asr
        # score = clean_acc - 0.5 * mean_asr  (penalize ASR but prioritize accuracy)
        score = metrics["clean_acc"] - 0.5 * metrics["mean_asr"]

        row = {
            "rank": 0,
            "ratio": cfg["ratio"],
            "h_scale": cfg["h_scale"],
            "m_scale": cfg["m_scale"],
            "bb_zoo": cfg["bb_zoo"],
            "bb_hsja": cfg["bb_hsja"],
            "clean_acc": round(metrics["clean_acc"], 2),
            "clean_f1": round(metrics["clean_f1"], 2),
            "mean_asr": round(metrics["mean_asr"], 2),
            "max_asr": round(metrics["max_asr"], 2),
            "score": round(score, 2),
            "time_s": round(elapsed, 1),
        }
        # Add per-attack ASRs
        for atk, asr in sorted(metrics["asrs"].items()):
            row[f"asr_{atk}"] = round(asr, 2)

        rows.append(row)

        marker = " ★" if score > best_score else ""
        if score > best_score:
            best_score = score
        print(
            f"[{i+1:3d}/{len(configs)}] "
            f"ratio={cfg['ratio']:.2f} h={cfg['h_scale']:.1f} m={cfg['m_scale']:.1f} "
            f"zoo={cfg['bb_zoo']:.0f} hsja={cfg['bb_hsja']:.0f} | "
            f"Acc={metrics['clean_acc']:.1f}% ASR_avg={metrics['mean_asr']:.1f}% "
            f"ASR_max={metrics['max_asr']:.1f}% "
            f"score={score:.1f} ({elapsed:.0f}s){marker}"
        )

    if not rows:
        print("No successful runs.")
        return

    # Sort by composite score
    rows.sort(key=lambda r: r["score"], reverse=True)
    for i, r in enumerate(rows):
        r["rank"] = i + 1

    # Save CSV
    output_path = os.path.join(_IEC, args.output)
    fieldnames = list(rows[0].keys())
    with open(output_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"\nResults saved to {output_path}")

    # Print top N
    print(f"\n{'='*80}")
    print(f"TOP {args.top} CONFIGURATIONS (score = clean_acc - 0.5 * mean_asr)")
    print(f"{'='*80}")
    print(f"{'Rank':>4} {'Ratio':>6} {'H':>4} {'M':>4} {'Zoo':>4} {'HSJA':>4} "
          f"{'Acc%':>6} {'F1%':>6} {'ASR_avg':>8} {'ASR_max':>8} {'Score':>6}")
    print("-" * 80)
    for r in rows[:args.top]:
        print(
            f"{r['rank']:4d} {r['ratio']:6.2f} {r['h_scale']:4.1f} {r['m_scale']:4.1f} "
            f"{r['bb_zoo']:4.0f} {r['bb_hsja']:4.0f} "
            f"{r['clean_acc']:6.1f} {r['clean_f1']:6.1f} "
            f"{r['mean_asr']:8.1f} {r['max_asr']:8.1f} {r['score']:6.1f}"
        )

    # Print best config as constants.py patch
    best = rows[0]
    print(f"\n{'='*80}")
    print("BEST CONFIG — paste into cli/utils/constants.py:")
    print(f"{'='*80}")
    base_weights = C.AT_WEIGHTS_ML.copy()
    # Reconstruct actual weights
    for k, v in orig["weights_ml"].items():
        if "deepfool" in k or "fgsm" in k:
            base_weights[k] = round(v * best["h_scale"], 1)
        elif "cw" in k or "mim" in k:
            base_weights[k] = round(v * best["m_scale"], 1)
        else:
            base_weights[k] = v

    print(f'AT_CLEAN_ADV_RATIO["rf"] = {best["ratio"]}')
    print(f"AT_WEIGHTS_ML = {base_weights}")
    print(f"AT_BB_WEIGHT_ZOO = {best['bb_zoo']}")
    print(f"AT_BB_WEIGHT_HSJA = {best['bb_hsja']}")


if __name__ == "__main__":
    main()
