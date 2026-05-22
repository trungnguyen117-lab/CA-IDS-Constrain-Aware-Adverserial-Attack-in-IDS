"""Quick CSFS ablation: train 5 tree models trên 4 variant feature-set.

Variant:
  A. RAW       — train_70_balanced_600_repro (80 feat, trước CSFS)
  B. COVAS     — drop 14 CovaS-dead features → 66 feat
  C. SHAP      — train_shap_58_600_repro (59 feat, sau CovaS+SHAP)
  D. SHAP+TVAE — modbus_train_merged_t1400_e200_repro (59 feat, +TVAE balance_to 1400)

Models: xgb, rf, cat, lgbm, et. Không ghi đè artifact nào ngoài tmp/csfs_ablation/.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[2]
DS = ROOT / "datasets"
OUT = Path(__file__).resolve().parent
LABEL_COL = "Label"
LABEL_NAMES = [
    "BENIGN", "BASELINE_REPLAY", "BRUTE_FORCE", "DELAY_RESPONSE",
    "FRAME_STACKING", "LENGTH_MANIPULATION", "PAYLOAD_INJECTION",
    "QUERY_FLOODING", "RECON",
]
N_CLASSES = 9
RNG = 42


def load(name: str) -> pd.DataFrame:
    return pd.read_csv(DS / name)


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


def make_xgb():
    return XGBClassifier(
        max_depth=8, n_estimators=3000, learning_rate=0.1,
        objective="multi:softprob", eval_metric="mlogloss",
        booster="gbtree", tree_method="hist", n_jobs=-1,
        random_state=RNG, early_stopping_rounds=50,
    )


def make_rf():
    return RandomForestClassifier(
        n_estimators=300, max_depth=20, criterion="entropy",
        class_weight="balanced", bootstrap=True, n_jobs=-1, random_state=RNG,
    )


def make_cat():
    return CatBoostClassifier(
        iterations=1500, depth=8, learning_rate=0.1,
        loss_function="MultiClass", eval_metric="MultiClass",
        random_seed=RNG, verbose=False, early_stopping_rounds=50,
        auto_class_weights="Balanced",
    )


def make_lgbm():
    return LGBMClassifier(
        n_estimators=2000, max_depth=-1, num_leaves=63, learning_rate=0.1,
        objective="multiclass", num_class=N_CLASSES, class_weight="balanced",
        n_jobs=-1, random_state=RNG, verbose=-1,
    )


def make_et():
    return ExtraTreesClassifier(
        n_estimators=300, max_depth=20, criterion="entropy",
        class_weight="balanced", bootstrap=False, n_jobs=-1, random_state=RNG,
    )


def fit_model(name: str, Xtr, ytr, Xte, yte):
    sw = compute_sample_weight("balanced", ytr)
    rng = np.random.RandomState(RNG)
    idx = rng.permutation(len(ytr))
    n_val = int(0.1 * len(ytr))
    iv, it = idx[:n_val], idx[n_val:]

    t0 = time.time()
    if name == "xgb":
        m = make_xgb()
        m.fit(Xtr[it], ytr[it], sample_weight=sw[it],
              eval_set=[(Xtr[iv], ytr[iv])],
              sample_weight_eval_set=[sw[iv]], verbose=False)
        best_iter = int(getattr(m, "best_iteration", -1))
    elif name == "cat":
        m = make_cat()
        m.fit(Xtr[it], ytr[it],
              eval_set=(Xtr[iv], ytr[iv]),
              use_best_model=True, verbose=False)
        best_iter = int(m.get_best_iteration() or -1)
    elif name == "lgbm":
        m = make_lgbm()
        m.fit(Xtr, ytr)
        best_iter = m.n_estimators_
    elif name == "rf":
        m = make_rf()
        m.fit(Xtr, ytr)
        best_iter = m.n_estimators
    elif name == "et":
        m = make_et()
        m.fit(Xtr, ytr)
        best_iter = m.n_estimators
    else:
        raise ValueError(name)
    fit_time = time.time() - t0

    ypred = np.asarray(m.predict(Xte)).reshape(-1)
    acc = accuracy_score(yte, ypred)
    f1m = f1_score(yte, ypred, average="macro")
    cm = confusion_matrix(yte, ypred, labels=list(range(N_CLASSES)))
    tpr, fpr = macro_tpr_fpr(cm)
    per = f1_score(yte, ypred, average=None, labels=list(range(N_CLASSES)))
    return dict(
        fit_time_sec=round(fit_time, 2),
        best_iter=best_iter,
        accuracy=float(acc),
        macro_f1=float(f1m),
        macro_tpr=tpr,
        macro_fpr=fpr,
        per_class_f1={LABEL_NAMES[i]: float(per[i]) for i in range(N_CLASSES)},
    )


def run_variant(name: str, train: pd.DataFrame, test: pd.DataFrame,
                models: list[str]) -> list[dict]:
    n_feat = train.shape[1] - 1
    print(f"\n=== {name} | n_features={n_feat} | "
          f"train={train.shape[0]} test={test.shape[0]} ===")
    Xtr = train.drop(columns=[LABEL_COL]).values
    ytr = train[LABEL_COL].values
    Xte = test.drop(columns=[LABEL_COL]).values
    yte = test[LABEL_COL].values

    out = []
    for mname in models:
        try:
            res = fit_model(mname, Xtr, ytr, Xte, yte)
        except Exception as e:
            print(f"  {mname}: FAILED {e!r}")
            continue
        res.update(variant=name, model=mname, n_features=n_feat)
        print(f"  {mname:5s} fit={res['fit_time_sec']:6.2f}s "
              f"acc={res['accuracy']:.4f}  f1m={res['macro_f1']:.4f}  "
              f"tpr={res['macro_tpr']:.4f}  fpr={res['macro_fpr']:.4f}")
        out.append(res)
    return out


def main() -> None:
    train_raw = load("train_70_balanced_600_repro.csv")
    test_raw  = load("test_30_600_repro.csv")
    train_shap = load("train_shap_58_600_repro.csv")
    test_shap  = load("test_shap_58_600_repro.csv")
    train_tvae = load("modbus_train_merged_t1400_e200_repro.csv")

    dead = json.loads(
        (ROOT / "report" / "covas_dead_features.json").read_text()
    )["dead_features"]
    print(f"CovaS dead features ({len(dead)}): {dead}")

    train_cov = train_raw.drop(columns=dead, errors="ignore")
    test_cov  = test_raw.drop(columns=dead, errors="ignore")

    models = ["xgb", "rf", "cat", "lgbm", "et"]
    all_results: list[dict] = []
    all_results += run_variant("A_RAW",     train_raw,  test_raw,  models)
    all_results += run_variant("B_COVAS",   train_cov,  test_cov,  models)
    all_results += run_variant("C_SHAP",    train_shap, test_shap, models)
    all_results += run_variant("D_SHAP_TVAE", train_tvae, test_shap, models)

    df = pd.DataFrame(all_results)

    print("\n=== Summary (all variants × models) ===")
    cols = ["variant", "model", "n_features", "fit_time_sec",
            "accuracy", "macro_f1", "macro_tpr", "macro_fpr"]
    print(df[cols].to_string(index=False))

    print("\n=== Macro-F1 pivot ===")
    pivot = df.pivot(index="model", columns="variant", values="macro_f1")
    pivot = pivot[["A_RAW", "B_COVAS", "C_SHAP", "D_SHAP_TVAE"]]
    print(pivot.round(4).to_string())

    (OUT / "results.json").write_text(json.dumps(all_results, indent=2, ensure_ascii=False))
    df[cols].to_csv(OUT / "summary.csv", index=False)
    pivot.round(4).to_csv(OUT / "macro_f1_pivot.csv")
    print(f"\n→ {OUT}/results.json + summary.csv + macro_f1_pivot.csv")


if __name__ == "__main__":
    main()
