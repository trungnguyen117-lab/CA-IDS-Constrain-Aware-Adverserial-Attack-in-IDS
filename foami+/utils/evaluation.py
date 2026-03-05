"""Evaluation helpers shared by the training and evaluate pipeline scripts."""
import logging
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix as sk_confusion_matrix

logger = logging.getLogger(__name__)


def macro_tpr_fpr(cm: np.ndarray):
    """Compute macro-averaged TPR and FPR from a confusion matrix.

    Returns:
        (macro_tpr, macro_fpr) as floats in [0, 1]
    """
    C = cm.shape[0]
    tprs, fprs = [], []
    for c in range(C):
        TP = cm[c, c]
        FN = cm[c, :].sum() - TP
        FP = cm[:, c].sum() - TP
        TN = cm.sum() - TP - FN - FP
        tprs.append(TP / (TP + FN) if (TP + FN) > 0 else 0.0)
        fprs.append(FP / (FP + TN) if (FP + TN) > 0 else 0.0)
    return float(np.mean(tprs)), float(np.mean(fprs))


def report_metrics(name: str, y_true: np.ndarray, y_pred: np.ndarray) -> None:
    """Log Accuracy, Macro-F1, Macro-TPR, Macro-FPR for a model."""
    acc = accuracy_score(y_true, y_pred)
    f1  = f1_score(y_true, y_pred, average='macro', zero_division=0)
    cm  = sk_confusion_matrix(y_true, y_pred)
    tpr, fpr = macro_tpr_fpr(cm)
    logger.info(
        f"[{name}] Acc={acc*100:.2f}%  Macro-F1={f1*100:.2f}%  "
        f"Macro-TPR={tpr*100:.2f}%  Macro-FPR={fpr*100:.4f}%"
    )


def predict_safe(predictor, X: np.ndarray, y_true: np.ndarray = None) -> np.ndarray:
    """Run predictor.predict(X) with NaN-sample handling.

    Samples containing NaN are treated as attack failures (predicted class 0).
    """
    has_nan = np.isnan(X).any(axis=1)
    n_nan = int(has_nan.sum())
    if n_nan > 0:
        logger.warning(f"[!] {n_nan}/{len(X)} NaN samples → treated as attack failure")

    y_pred = np.zeros(len(X), dtype=np.int64)
    valid = ~has_nan
    if valid.any():
        proba = np.asarray(predictor.predict(X[valid].astype(np.float32)))
        y_pred[valid] = proba.argmax(axis=1) if proba.ndim == 2 else proba
    return y_pred


def asr(y_plain_pred: np.ndarray,
        y_adv_pred: np.ndarray,
        y_true: np.ndarray) -> float:
    """Attack Success Rate: fraction of originally-correct samples now misclassified."""
    n = min(len(y_plain_pred), len(y_adv_pred), len(y_true))
    correct = y_plain_pred[:n] == y_true[:n]
    denom = int(correct.sum())
    if denom == 0:
        return 0.0
    return float(((y_adv_pred[:n] != y_true[:n]) & correct).sum()) / denom


def format_cm(cm: np.ndarray, class_names: list) -> str:
    """Return a text-formatted confusion matrix (rows=true, cols=pred)."""
    col_w   = max(max(len(c) for c in class_names),
                  max(len(str(v)) for v in cm.flatten())) + 2
    label_w = max(len(c) for c in class_names) + 1
    header  = ' ' * label_w + ''.join(c.rjust(col_w) for c in class_names)
    lines   = ['Confusion matrix (rows=true, cols=pred):', header]
    for i, row in enumerate(cm):
        lines.append(class_names[i].rjust(label_w) +
                     ''.join(str(v).rjust(col_w) for v in row))
    return '\n'.join(lines)


def save_cm_plot(cm: np.ndarray, class_names: list,
                 title: str, out_path: str) -> None:
    """Save a confusion-matrix heatmap PNG to *out_path*."""
    n = len(class_names)
    fig, ax = plt.subplots(figsize=(max(5, n), max(4, n - 1)))
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(class_names, rotation=45, ha='right', fontsize=9)
    ax.set_yticklabels(class_names, fontsize=9)
    ax.set_xlabel('Predicted label', fontsize=10)
    ax.set_ylabel('True label', fontsize=10)
    ax.set_title(title, fontsize=11, pad=12)
    thresh = cm.max() / 2.0
    for i in range(n):
        for j in range(n):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center', fontsize=9,
                    color='white' if cm[i, j] > thresh else 'black')
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info(f"    [CM] Saved: {out_path}")
