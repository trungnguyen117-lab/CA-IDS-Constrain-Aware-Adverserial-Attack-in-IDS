"""CSFS ablation — load đúng cấu hình từ config/training/*.yaml + DNN.

Variants (giống run_xgb_ablation.py):
  A_RAW       — train_70_balanced_600_repro (80 feat)
  B_COVAS     — drop 14 CovaS-dead features → 66 feat
  C_SHAP      — train_shap_58_600_repro (58 feat)
  D_SHAP_TVAE — modbus_train_merged_t1400_e200_repro (58 feat, +TVAE 1400/class)

Models:
  - xgb, rf, cat, lgbm, et  → src.core.models.* (đọc config/training/<m>.yaml).
  - dnn                     → model_local.DNNModel.DEFAULT_CFG.

Output → tmp/csfs_ablation/full_{results.json,summary.csv,macro_f1_pivot.csv}.
Không đè artifact pipeline (baseline/models/, defense/, datasets/, report/).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.utils.class_weight import compute_sample_weight

from src.core.models import (
    CatBoostModel, ExtraTreesModel, LightGBMModel, RandomForestModel,
    XGBoostModel,
)

ROOT = Path(__file__).resolve().parents[2]
DS = ROOT / "datasets"
CFG_TRAIN = ROOT / "config" / "training"
OUT = Path(__file__).resolve().parent
LABEL_COL = "Label"
LABEL_NAMES = [
    "BENIGN", "BASELINE_REPLAY", "BRUTE_FORCE", "DELAY_RESPONSE",
    "FRAME_STACKING", "LENGTH_MANIPULATION", "PAYLOAD_INJECTION",
    "QUERY_FLOODING", "RECON",
]
N_CLASSES = 9
RNG = 42

TREE_CLS = {
    "xgb": XGBoostModel, "rf": RandomForestModel, "cat": CatBoostModel,
    "lgbm": LightGBMModel, "et": ExtraTreesModel,
}


def load_csv(name: str) -> pd.DataFrame:
    return pd.read_csv(DS / name)


def load_yaml(model_name: str) -> dict:
    p = CFG_TRAIN / f"{model_name}.yaml"
    return yaml.safe_load(p.read_text()) or {}


def macro_tpr_fpr(cm: np.ndarray) -> tuple[float, float]:
    K = cm.shape[0]
    tpr, fpr = [], []
    total = cm.sum()
    for i in range(K):
        TP = cm[i, i]
        FN = cm[i, :].sum() - TP
        FP = cm[:, i].sum() - TP
        TN = total - TP - FN - FP
        tpr.append(TP / (TP + FN) if (TP + FN) else 0.0)
        fpr.append(FP / (FP + TN) if (FP + TN) else 0.0)
    return float(np.mean(tpr)), float(np.mean(fpr))


def split_val(X, y, val_size=0.1, seed=RNG):
    sss = StratifiedShuffleSplit(n_splits=1, test_size=val_size, random_state=seed)
    tr, va = next(sss.split(X, y))
    return X[tr], y[tr], X[va], y[va]


def fit_tree(name: str, Xtr, ytr, Xte, yte) -> dict:
    cfg = load_yaml(name)
    cls = TREE_CLS[name]
    needs_val = "early_stopping_rounds" in cfg  # xgb/cat/lgbm

    t0 = time.time()
    if needs_val:
        Xt, yt, Xv, yv = split_val(Xtr, ytr)
        sw_tr = compute_sample_weight("balanced", yt)
        sw_val = compute_sample_weight("balanced", yv)
        if name == "xgb":
            m = cls().train(Xt, yt, X_val=Xv, y_val=yv, cfg=cfg,
                            sample_weight=sw_tr, sample_weight_eval=sw_val)
        else:
            m = cls().train(Xt, yt, X_val=Xv, y_val=yv, cfg=cfg)
    else:
        m = cls().train(Xtr, ytr, cfg=cfg)
    fit_time = time.time() - t0

    ypred = np.asarray(m.predict(Xte)).reshape(-1)
    return _metrics(ytr, yte, ypred, fit_time, cfg=cfg)


def fit_dnn(Xtr, ytr, Xte, yte, device="cpu") -> dict:
    from model_local.dnn import DNNModel
    Xt, yt, Xv, yv = split_val(Xtr.astype(np.float32), ytr)
    t0 = time.time()
    m = DNNModel().train(Xt, yt, X_val=Xv, y_val=yv, device=device)
    fit_time = time.time() - t0
    ypred = m.predict(Xte.astype(np.float32))
    return _metrics(ytr, yte, ypred, fit_time, cfg=m._cfg)


def _metrics(ytr, yte, ypred, fit_time, cfg) -> dict:
    acc = accuracy_score(yte, ypred)
    f1m = f1_score(yte, ypred, average="macro")
    cm = confusion_matrix(yte, ypred, labels=list(range(N_CLASSES)))
    tpr, fpr = macro_tpr_fpr(cm)
    per = f1_score(yte, ypred, average=None, labels=list(range(N_CLASSES)))
    return dict(
        fit_time_sec=round(fit_time, 2),
        accuracy=float(acc),
        macro_f1=float(f1m),
        macro_tpr=tpr,
        macro_fpr=fpr,
        per_class_f1={LABEL_NAMES[i]: float(per[i]) for i in range(N_CLASSES)},
        cfg_used={k: v for k, v in cfg.items()
                  if isinstance(v, (int, float, str, bool, list, tuple))},
    )


def run_variant(name: str, train: pd.DataFrame, test: pd.DataFrame,
                models: list[str], device: str) -> list[dict]:
    n_feat = train.shape[1] - 1
    print(f"\n=== {name} | n_features={n_feat} | "
          f"train={train.shape[0]} test={test.shape[0]} ===")
    Xtr = train.drop(columns=[LABEL_COL]).values
    ytr = train[LABEL_COL].values.astype(int)
    Xte = test.drop(columns=[LABEL_COL]).values
    yte = test[LABEL_COL].values.astype(int)

    out = []
    for mname in models:
        try:
            if mname == "dnn":
                res = fit_dnn(Xtr, ytr, Xte, yte, device=device)
            else:
                res = fit_tree(mname, Xtr, ytr, Xte, yte)
        except Exception as e:
            print(f"  {mname}: FAILED {e!r}")
            continue
        res.update(variant=name, model=mname, n_features=n_feat)
        print(f"  {mname:5s} fit={res['fit_time_sec']:7.2f}s "
              f"acc={res['accuracy']:.4f}  f1m={res['macro_f1']:.4f}  "
              f"tpr={res['macro_tpr']:.4f}  fpr={res['macro_fpr']:.4f}")
        out.append(res)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="+",
                        default=["xgb", "rf", "cat", "lgbm", "et", "dnn"],
                        choices=["xgb", "rf", "cat", "lgbm", "et", "dnn"])
    parser.add_argument("--variants", nargs="+",
                        default=["A_RAW", "B_COVAS", "C_SHAP", "D_SHAP_TVAE"],
                        choices=["A_RAW", "B_COVAS", "C_SHAP", "D_SHAP_TVAE"])
    parser.add_argument("--device", default="cpu",
                        help="torch device for DNN (cpu|cuda)")
    args = parser.parse_args()

    train_raw = load_csv("train_70_balanced_600_repro.csv")
    test_raw = load_csv("test_30_600_repro.csv")
    train_shap = load_csv("train_shap_58_600_repro.csv")
    test_shap = load_csv("test_shap_58_600_repro.csv")
    train_tvae = load_csv("modbus_train_merged_t1400_e800_repro.csv")

    dead = json.loads(
        (ROOT / "report" / "covas_dead_features.json").read_text()
    )["dead_features"]
    print(f"CovaS dead features ({len(dead)}): {dead}")

    train_cov = train_raw.drop(columns=dead, errors="ignore")
    test_cov = test_raw.drop(columns=dead, errors="ignore")

    sources = {
        "A_RAW":      (train_raw, test_raw),
        "B_COVAS":    (train_cov, test_cov),
        "C_SHAP":     (train_shap, test_shap),
        "D_SHAP_TVAE": (train_tvae, test_shap),
    }

    all_results: list[dict] = []
    for v in args.variants:
        tr, te = sources[v]
        all_results += run_variant(v, tr, te, args.models, args.device)

    df = pd.DataFrame(all_results)

    print("\n=== Summary (all variants × models) ===")
    cols = ["variant", "model", "n_features", "fit_time_sec",
            "accuracy", "macro_f1", "macro_tpr", "macro_fpr"]
    print(df[cols].to_string(index=False))

    print("\n=== Macro-F1 pivot ===")
    pivot = df.pivot(index="model", columns="variant", values="macro_f1")
    pivot = pivot[[c for c in ["A_RAW", "B_COVAS", "C_SHAP", "D_SHAP_TVAE"]
                   if c in pivot.columns]]
    print(pivot.round(4).to_string())

    (OUT / "full_results.json").write_text(
        json.dumps(all_results, indent=2, ensure_ascii=False))
    df[cols].to_csv(OUT / "full_summary.csv", index=False)
    pivot.round(4).to_csv(OUT / "full_macro_f1_pivot.csv")
    print(f"\n→ {OUT}/full_results.json + full_summary.csv + full_macro_f1_pivot.csv")


if __name__ == "__main__":
    main()
