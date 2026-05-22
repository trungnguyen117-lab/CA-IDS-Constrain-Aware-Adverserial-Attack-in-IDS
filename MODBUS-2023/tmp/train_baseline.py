"""Smoke-test train baseline trên data repro (TVAE-augmented).

Train RandomForest + XGBoost, đánh giá macro-F1 + classification_report,
dump model .pkl ra tmp/.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parent.parent
TRAIN_CSV = ROOT / "datasets" / "modbus_train_merged_t1400_e200_repro.csv"
TEST_CSV = ROOT / "datasets" / "test_shap_58_600_repro.csv"
OUT_DIR = ROOT / "tmp" / "models"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def evaluate(name: str, model, X_test, y_test) -> float:
    t0 = time.time()
    pred = model.predict(X_test)
    dt = time.time() - t0
    macro_f1 = f1_score(y_test, pred, average="macro")
    print(f"\n{'=' * 60}\n{name}  |  predict {dt:.3f}s  |  macro-F1 {macro_f1:.4f}\n{'=' * 60}")
    print(classification_report(y_test, pred, digits=4))
    return macro_f1


def main() -> None:
    print(f"Train: {TRAIN_CSV.name}")
    print(f"Test : {TEST_CSV.name}")
    train = pd.read_csv(TRAIN_CSV)
    test = pd.read_csv(TEST_CSV)
    print(f"Train shape: {train.shape}  | Test shape: {test.shape}")

    X_train, y_train = train.drop(columns=["Label"]), train["Label"]
    X_test, y_test = test.drop(columns=["Label"]), test["Label"]

    rf = RandomForestClassifier(
        n_estimators=908, max_leaf_nodes=15000, criterion="entropy",
        class_weight="balanced", bootstrap=True, n_jobs=-1, random_state=0,
    )
    t0 = time.time()
    rf.fit(X_train, y_train)
    print(f"\n[RF fit] {time.time() - t0:.1f}s")
    rf_f1 = evaluate("RandomForest", rf, X_test, y_test)
    joblib.dump(rf, OUT_DIR / "rf_baseline.pkl")

    sw = compute_sample_weight("balanced", y_train)
    xgb = XGBClassifier(
        max_depth=15, n_estimators=300, learning_rate=0.2,
        objective="multi:softprob", num_class=y_train.nunique(),
        booster="gbtree", tree_method="hist", n_jobs=-1, random_state=42,
        eval_metric="mlogloss",
    )
    t0 = time.time()
    xgb.fit(X_train, y_train, sample_weight=sw)
    print(f"\n[XGB fit] {time.time() - t0:.1f}s")
    xgb_f1 = evaluate("XGBoost", xgb, X_test, y_test)
    joblib.dump(xgb, OUT_DIR / "xgb_baseline.pkl")

    print(f"\n{'=' * 60}\nSummary | RF macro-F1: {rf_f1:.4f}  |  XGB macro-F1: {xgb_f1:.4f}")
    print(f"Models dumped to: {OUT_DIR}")


if __name__ == "__main__":
    main()
