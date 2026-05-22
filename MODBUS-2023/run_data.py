"""End-to-end preprocessing pipeline for MODBUS-2023.

Full stages (notebook 0+1 logic + CovaS/SHAP/TVAE):

    clean → split → balance → covas → shap → apply → topup

Stages ``clean / split / balance`` reproduce notebook 0+1 via
``MODBUS-2023/src/modbus_prep.py`` (MODBUS-specific recipe). Stages
``covas / shap / apply / topup`` reuse ``src/core/preprocessing.py``
shared with IEC-104. All output files carry ``_repro`` suffix so they do
not overwrite canonical artifacts.

Hyperparameters are read from ``MODBUS-2023/config.yaml`` (or a sibling
``config.smoke.yaml`` via ``MODBUS_CONFIG=...``). Nothing is hardcoded here.
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

_CFG_FILE = os.environ.get("MODBUS_CONFIG", "config.yaml")
SPEC = Config.from_yaml(_ROOT / _CFG_FILE)

# Local MODBUS-specific recipe (notebook 0+1).
from modbus_prep import (  # noqa: E402
    ModbusCleaner, ModbusRecipe, ModbusSplitter, class_dist,
)

logger = logging.getLogger("modbus.run_data")

STAGES = ("clean", "split", "balance", "covas", "shap", "apply", "topup")


def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )


def topup_path(target: int, epochs: int) -> Path:
    base = SPEC.paths.get("topup_dir") or (SPEC.root / "datasets")
    return Path(base) / f"modbus_train_merged_t{target}_e{epochs}_repro.csv"


def report_path(name: str) -> Path:
    out = SPEC.paths.report
    out.mkdir(parents=True, exist_ok=True)
    return out / name


def stage_idx(name: str) -> int:
    if name not in STAGES:
        raise ValueError(f"Unknown stage {name!r}; valid: {STAGES}")
    return STAGES.index(name)


def shap_input_paths(start: int) -> tuple[Path, Path]:
    """Khi pipeline chạy từ clean/split/balance, dùng output _repro;
    khi --from covas (legacy), dùng canonical."""
    paths = SPEC.paths
    if start <= stage_idx("balance"):
        return paths.get("train_for_shap_repro"), paths.get("test_for_shap_repro")
    return paths.get("train_for_shap"), paths.get("test_for_shap")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--from", dest="from_stage", default="clean",
                        choices=STAGES, help="first stage to run")
    parser.add_argument("--until", default="topup", choices=STAGES,
                        help="last stage to run (inclusive)")
    parser.add_argument("--target", type=int, default=1400,
                        help="per-class target after balance_to (or rare-class "
                             "target for topup_rare); matches canonical=1400")
    parser.add_argument("--epochs", type=int, default=200,
                        help="TVAE epochs override")
    parser.add_argument("--mode", choices=("balance_to", "topup_rare"),
                        default=None,
                        help="Augment mode; default reads cfg.augment.mode "
                             "(MODBUS uses balance_to)")
    parser.add_argument("--top-k", type=int, default=None,
                        help="Override SHAP top_k (no yaml change)")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    setup_logging(args.log_level)

    SPEC.augment.setdefault("tvae", {})["epochs"] = args.epochs
    if args.top_k is not None:
        SPEC.shap["top_k"] = args.top_k

    prep = PrepareData(SPEC)
    aug = DataAugmentation(SPEC)
    paths = SPEC.paths
    recipe = ModbusRecipe.from_config(SPEC)

    start = stage_idx(args.from_stage)
    end = stage_idx(args.until)
    if end < start:
        raise SystemExit(f"--until ({args.until}) before --from ({args.from_stage})")

    logger.info("Pipeline %s → %s | target=%d | epochs=%d",
                STAGES[start], STAGES[end], args.target, args.epochs)

    # --- clean ---
    if start <= stage_idx("clean") <= end:
        raw_path = paths.get("raw")
        if raw_path is None:
            raise SystemExit("paths.raw not set in config.yaml")
        logger.info("Reading raw %s (this may take a while for ~5GB)", raw_path)
        df_raw = pd.read_csv(raw_path, low_memory=False)
        df_clean, rep = ModbusCleaner(recipe).clean(df_raw)
        out = paths.get("cleaned_repro")
        out.parent.mkdir(parents=True, exist_ok=True)
        df_clean.to_csv(out, index=False)
        report_path("clean_summary.json").write_text(json.dumps({
            "raw": str(raw_path),
            "output": str(out),
            "shape": [rep.n_rows, rep.n_cols],
            "constant_dropped": rep.constant_dropped,
            "manual_dropped": rep.manual_dropped,
            "label_mapping": rep.label_mapping,
            "label_dist": class_dist(df_clean, recipe.label_col),
        }, indent=2, ensure_ascii=False))
        logger.info("→ %s  shape=%s", out, df_clean.shape)

    # --- split ---
    if start <= stage_idx("split") <= end:
        df = pd.read_csv(paths.get("cleaned_repro"))
        train_df, test_df = ModbusSplitter(recipe).split(df)
        train_out = paths.get("train_split_repro")
        test_out = paths.get("test_for_shap_repro")
        train_out.parent.mkdir(parents=True, exist_ok=True)
        test_out.parent.mkdir(parents=True, exist_ok=True)
        train_df.to_csv(train_out, index=False)
        test_df.to_csv(test_out, index=False)
        report_path("split_summary.json").write_text(json.dumps({
            "train_out": str(train_out),
            "test_out": str(test_out),
            "train_shape": list(train_df.shape),
            "test_shape": list(test_df.shape),
            "train_label_dist": class_dist(train_df, recipe.label_col),
            "test_label_dist": class_dist(test_df, recipe.label_col),
        }, indent=2, ensure_ascii=False))
        logger.info("→ train=%s  test=%s", train_df.shape, test_df.shape)

    # --- balance ---
    if start <= stage_idx("balance") <= end:
        df = pd.read_csv(paths.get("train_split_repro"))
        balanced = ModbusSplitter(recipe).balance(df)
        out = paths.get("train_for_shap_repro")
        out.parent.mkdir(parents=True, exist_ok=True)
        balanced.to_csv(out, index=False)
        report_path("balance_summary.json").write_text(json.dumps({
            "output": str(out),
            "shape": list(balanced.shape),
            "label_dist": class_dist(balanced, recipe.label_col),
        }, indent=2, ensure_ascii=False))
        logger.info("→ %s  shape=%s", out, balanced.shape)

    # Determine input source for downstream SHAP stages (auto-pick _repro vs canonical)
    train_in, test_in = shap_input_paths(start)

    # --- covas ---
    if start <= stage_idx("covas") <= end:
        df = pd.read_csv(train_in)
        dead, stats = prep.covas_dead(df)
        stats.to_csv(report_path("covas_stats.csv"), index=False)
        report_path("covas_dead_features.json").write_text(json.dumps({
            "pair": list(SPEC.covas.get("pair", ())),
            "thresholds": SPEC.covas.get("thresholds", {}),
            "dead_features": dead,
            "n_features_total": int(df.shape[1] - 1),
            "n_features_dead": len(dead),
        }, indent=2, ensure_ascii=False))
        logger.info("→ %s  dead=%d",
                    report_path("covas_dead_features.json"), len(dead))

    # --- shap ---
    if start <= stage_idx("shap") <= end:
        df = pd.read_csv(train_in)
        dead = json.loads(report_path("covas_dead_features.json").read_text())["dead_features"]
        use_csv = SPEC.shap.get("use_features_csv")
        if use_csv:
            csv_path = SPEC.resolve(use_csv)
            selected = sorted(
                line.strip() for line in csv_path.read_text().splitlines() if line.strip()
            )
            n_candidate = int(df.shape[1] - 1 - len(dead))
            logger.info("Using pre-computed selected features from %s (%d features)",
                        csv_path, len(selected))
        else:
            selected, ranking = prep.shap_select(df, drop=tuple(dead))
            ranking.to_csv(report_path("shap_ranking.csv"), index=False)
            n_candidate = int(df.shape[1] - 1 - len(dead))
        report_path("shap_selected_features.json").write_text(json.dumps({
            "source": "csv" if use_csv else "recompute",
            "top_k": SPEC.shap.get("top_k", 50),
            "n_candidate": n_candidate,
            "n_selected": len(selected),
            "selected": selected,
            "covas_dead": dead,
        }, indent=2, ensure_ascii=False))
        logger.info("→ %s  selected=%d",
                    report_path("shap_selected_features.json"), len(selected))

    # --- apply ---
    if start <= stage_idx("apply") <= end:
        selected = json.loads(
            report_path("shap_selected_features.json").read_text())["selected"]
        train = pd.read_csv(train_in)
        test = pd.read_csv(test_in)
        train_shap = prep.apply_features(train, selected)
        test_shap = prep.apply_features(test, selected)
        train_out = paths.get("train_shap_real")
        test_out = paths.get("test_shap_real")
        train_out.parent.mkdir(parents=True, exist_ok=True)
        test_out.parent.mkdir(parents=True, exist_ok=True)
        train_shap.to_csv(train_out, index=False)
        test_shap.to_csv(test_out, index=False)
        logger.info("→ %s  train=%s  test=%s",
                    train_out, train_shap.shape, test_shap.shape)

    # --- topup ---
    if start <= stage_idx("topup") <= end:
        df = pd.read_csv(paths.get("train_shap_real"))
        mode = args.mode or SPEC.augment.get("mode", "topup_rare")
        if mode == "balance_to":
            out, synth_counts = aug.balance_to(df, target=args.target)
        elif mode == "topup_rare":
            out, synth_counts = aug.topup_rare(df, target_rare=args.target)
        else:
            raise SystemExit(f"Unknown augment mode: {mode!r}")
        out_path = topup_path(args.target, args.epochs)
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


if __name__ == "__main__":
    main()
