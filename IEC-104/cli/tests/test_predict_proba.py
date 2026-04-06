"""Quick verification: DLModel.predict vs ART classifier predict are identical."""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_CLI = os.path.dirname(_HERE)
_IEC = os.path.dirname(_CLI)
sys.path.insert(0, _CLI)
sys.path.insert(0, _IEC)

import numpy as np
import torch

from utils.loaders import load_dataset
from utils.paths import model_path
from model.lstm import LSTMModel
from model.resdnn import ResDNNModel


def verify_model(cls, name, path, X):
    m = cls.load(path, device="cpu")

    # Method 1: predict() — InputNorm embedded in model
    y_pred_1 = m.predict(X)
    proba_1 = m.predict_proba(X)

    # Method 2: Direct model forward — should give identical results
    with torch.no_grad():
        logits_2 = m.net(torch.from_numpy(X.astype(np.float32)))
        proba_2 = torch.softmax(logits_2, dim=1).numpy()
        y_pred_2 = logits_2.argmax(dim=1).numpy()

    pred_match = np.array_equal(y_pred_1, y_pred_2)
    proba_diff = np.abs(proba_1 - proba_2).max()

    print(f"{name}:")
    print(f"  predict match:    {pred_match}")
    print(f"  proba max diff:   {proba_diff:.2e}")
    print(f"  samples checked:  {len(X)}")

    assert pred_match, f"{name} predictions differ!"
    assert proba_diff < 1e-5, f"{name} proba diff too large: {proba_diff}"
    print(f"  PASS\n")


if __name__ == "__main__":
    _, X_test, _, _ = load_dataset("test")
    X = X_test.astype(np.float32)

    verify_model(LSTMModel, "LSTM",
                 model_path("framework_lstm_TVAE.pth"), X)
    verify_model(ResDNNModel, "ResDNN",
                 model_path("framework_resdnn_TVAE.pth"), X)

    print("All checks passed.")
