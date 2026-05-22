"""End-to-end data preprocessing + augmentation pipeline for IIOT-2021.

Stages run sequentially (no skip / partial run):
  preprocess  raw → clean
  split       clean → train_real + test_real
  cap         train_real → train_for_shap
  covas       train_for_shap → CovaS dead-features JSON
  shap        train_for_shap (− dead) → SHAP selected features JSON
  apply       train_for_shap + test_real + selected → train_shap_real + test_final
  topup       train_shap_real → train_topup_t<T>_e<E>.csv (TVAE rare augment)

All transforms live in ``src/core/preprocessing.py``; this script only
parses CLI flags, reads/writes CSVs, and writes JSON reports. Every
hyperparameter is read from ``IIOT-2021/config.yaml``.

Override config:  IIOT_CONFIG=config.smoke.yaml python run_data.py
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))

import pandas as pd  # noqa: E402

from src.core.preprocessing import DataAugmentation, PrepareData  # noqa: E402
from src.core.config import Config  # noqa: E402

_CFG_FILE = os.environ.get("IIOT_CONFIG", "config.yaml")
SPEC = Config.from_yaml(_ROOT / _CFG_FILE)

logger = logging.getLogger("iiot.run_data")


def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )


def topup_path(target: int, epochs: int, mode: str = "topup") -> Path:
    base = SPEC.paths.get("topup_dir") or (SPEC.root / "datasets")
    prefix = "train_balance" if mode == "balance" else "train_topup"
    suffix = "_repro" if os.environ.get("IIOT_CONFIG", "").endswith("config_v2.yaml") else ""
    return Path(base) / f"{prefix}_t{target}_e{epochs}{suffix}.csv"


def report_path(name: str) -> Path:
    out = SPEC.paths.report
    out.mkdir(parents=True, exist_ok=True)
    return out / name


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--target", type=int, default=1000,
                        help="per-class target (topup: rare only; balance: every class)")
    parser.add_argument("--mode", choices=("topup", "balance"), default="topup",
                        help="topup_rare (rare classes only) or balance_to (every class)")
    parser.add_argument("--epochs", type=int, default=50,
                        help="TVAE epochs override")
    parser.add_argument("--top-k", type=int, default=None,
                        help="Override SHAP top_k (no yaml change)")
    parser.add_argument("--cap-method", default="random",
                        choices=("random", "minibatch-kmeans"))
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    setup_logging(args.log_level)

    SPEC.augment.setdefault("tvae", {})["epochs"] = args.epochs
    if args.top_k is not None:
        SPEC.shap["top_k"] = args.top_k

    prep = PrepareData(SPEC)
    aug = DataAugmentation(SPEC)
    paths = SPEC.paths

    logger.info("Pipeline preprocess → topup | target=%d | epochs=%d",
                args.target, args.epochs)

    # --- preprocess (raw → clean) ---
    df_raw = pd.read_csv(paths.get("raw"))
    df_clean, mapping = prep.clean(df_raw)
    df_clean, dropped = prep.drop_constants(df_clean)
    if dropped:
        logger.info("Dropped %d constant features: %s", len(dropped), dropped)
        report_path("dropped_constants.json").write_text(
            json.dumps({"dropped": dropped}, indent=2, ensure_ascii=False))
    df_clean, n_dup = prep.drop_dup_features(df_clean)
    if n_dup:
        logger.info("Dropped %d duplicate-feature rows (anti-leak)", n_dup)
        report_path("dropped_duplicates.json").write_text(json.dumps({
            "dropped_dup_rows": int(n_dup),
            "remaining": int(len(df_clean)),
        }, indent=2))
    df_clean.to_csv(paths.get("clean"), index=False)
    if mapping:
        report_path("traffic_label_mapping.json").write_text(
            json.dumps(mapping, indent=2, ensure_ascii=False))
    logger.info("→ %s", paths.get("clean"))

    # --- split ---
    train_df, test_df = prep.split(df_clean)
    train_df.to_csv(paths.get("train_real"), index=False)
    test_df.to_csv(paths.get("test_real"), index=False)
    logger.info("→ %s  train=%s test=%s",
                paths.get("train_real"), train_df.shape, test_df.shape)

    # --- cap ---
    train_capped = prep.cap(train_df, method=args.cap_method)
    train_capped.to_csv(paths.get("train_for_shap"), index=False)
    logger.info("→ %s  shape=%s", paths.get("train_for_shap"), train_capped.shape)

    # --- covas ---
    dead, stats = prep.covas_dead(train_capped)
    stats.to_csv(report_path("covas_stats.csv"), index=False)
    report_path("covas_dead_features.json").write_text(json.dumps({
        "pair": list(SPEC.covas.get("pair", ())),
        "thresholds": SPEC.covas.get("thresholds", {}),
        "dead_features": dead,
        "n_features_total": int(train_capped.shape[1] - 1),
        "n_features_dead": len(dead),
    }, indent=2, ensure_ascii=False))
    logger.info("→ %s  dead=%d",
                report_path("covas_dead_features.json"), len(dead))

    # --- shap ---
    selected, ranking = prep.shap_select(train_capped, drop=tuple(dead))
    ranking.to_csv(report_path("shap_ranking.csv"), index=False)
    report_path("shap_selected_features.json").write_text(json.dumps({
        "top_k": SPEC.shap.get("top_k", 30),
        "n_candidate": int(train_capped.shape[1] - 1 - len(dead)),
        "n_selected": len(selected),
        "selected": selected,
        "covas_dead": dead,
    }, indent=2, ensure_ascii=False))
    logger.info("→ %s  selected=%d",
                report_path("shap_selected_features.json"), len(selected))

    # --- apply ---
    train_shap = prep.apply_features(train_capped, selected)
    test_final = prep.apply_features(test_df, selected)
    train_shap.to_csv(paths.get("train_shap_real"), index=False)
    test_out = paths.get("test_final") or paths.test
    test_final.to_csv(test_out, index=False)
    logger.info("→ %s  train=%s test=%s",
                paths.get("train_shap_real"),
                train_shap.shape, test_final.shape)

    # --- augment (topup_rare or balance_to) ---
    if args.mode == "balance":
        out, synth_counts = aug.balance_to(train_shap, target=args.target)
    else:
        out, synth_counts = aug.topup_rare(train_shap, target_rare=args.target)
    out_path = topup_path(args.target, args.epochs, mode=args.mode)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    report_path(f"topup_t{args.target}_e{args.epochs}.json").write_text(
        json.dumps({
            "target_rare": args.target,
            "epochs": args.epochs,
            "synth_per_class": synth_counts,
            "shape": list(out.shape),
            "label_dist": out[SPEC.label_col].value_counts().sort_index().to_dict(),
            "output": str(out_path),
        }, indent=2, ensure_ascii=False))
    logger.info("→ %s  shape=%s synth=%s", out_path, out.shape, synth_counts)

    # --- validate cont_features ---
    if SPEC.cont_features:
        final_cols = set(out.columns)
        declared = set(SPEC.cont_features)
        present = sorted(declared & final_cols)
        missing = sorted(declared - final_cols)
        if missing:
            logger.warning("cont_features missing in final CSV: %s", missing)
        report_path("cont_features_resolved.json").write_text(json.dumps({
            "declared": sorted(declared),
            "present": present,
            "missing": missing,
        }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
