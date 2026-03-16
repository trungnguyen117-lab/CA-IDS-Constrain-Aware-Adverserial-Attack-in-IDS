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


# ── Evaluator class ───────────────────────────────────────────────────────────

class Evaluator:
    """Evaluation toolkit for model predictions.

    Usage:
        ev = Evaluator(class_names=['benign', 'attack1', ...], report_dir='report/')
        ev.report('LSTM-baseline', y_true, y_pred)
        rate = ev.asr(y_plain_pred, y_adv_pred, y_true)
        ev.confusion_matrix('LSTM vs PGD', y_true, y_pred, save_plot=True)
    """

    def __init__(self, class_names=None, report_dir=None):
        self.class_names = class_names
        self.report_dir = report_dir

    def report(self, name, y_true, y_pred):
        """Log Accuracy, Macro-F1, Macro-TPR, Macro-FPR."""
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
        cm = sk_confusion_matrix(y_true, y_pred)
        tpr, fpr = macro_tpr_fpr(cm)
        logger.info(
            f"[{name}] Acc={acc*100:.2f}%  Macro-F1={f1*100:.2f}%  "
            f"Macro-TPR={tpr*100:.2f}%  Macro-FPR={fpr*100:.4f}%"
        )
        return {'accuracy': acc, 'f1': f1, 'tpr': tpr, 'fpr': fpr}

    @staticmethod
    def asr(y_plain_pred, y_adv_pred, y_true):
        """Attack Success Rate: fraction of originally-correct samples now misclassified."""
        n = min(len(y_plain_pred), len(y_adv_pred), len(y_true))
        correct = y_plain_pred[:n] == y_true[:n]
        denom = int(correct.sum())
        if denom == 0:
            return 0.0
        return float(((y_adv_pred[:n] != y_true[:n]) & correct).sum()) / denom

    @staticmethod
    def predict_safe(predictor, X, y_true=None):
        """Run predictor.predict(X) with NaN-sample handling."""
        has_nan = np.isnan(X).any(axis=1)
        n_nan = int(has_nan.sum())
        if n_nan > 0:
            logger.warning(f"[!] {n_nan}/{len(X)} NaN samples → treated as attack failure")
        y_pred = np.zeros(len(X), dtype=np.int64)
        valid = ~has_nan
        if valid.any():
            y_pred[valid] = np.asarray(predictor.predict(X[valid].astype(np.float32)))
        return y_pred

    def confusion_matrix(self, title, y_true, y_pred, save_plot=False):
        """Format and optionally save confusion matrix."""
        cm = sk_confusion_matrix(y_true, y_pred)
        text = format_cm(cm, self.class_names) if self.class_names else str(cm)
        logger.info(f"\n{text}")
        if save_plot and self.class_names and self.report_dir:
            out = os.path.join(self.report_dir, f"{title.replace(' ', '_')}_cm.png")
            save_cm_plot(cm, self.class_names, title, out)
        return cm

    # ── Result table formatting ───────────────────────────────────────────────

    @staticmethod
    def results_table(results):
        """Build a detailed PrettyTable from a list of result dicts.

        Each dict must have: target, attack, plain_acc, plain_f1,
        adv_acc, adv_f1, asr.
        """
        from prettytable import PrettyTable
        tbl = PrettyTable(['Target', 'Attack', 'Plain Acc', 'Plain F1',
                           'Adv Acc', 'Adv F1', 'ASR'])
        tbl.align = 'r'
        tbl.align['Target'] = 'l'
        tbl.align['Attack'] = 'l'
        for r in results:
            tbl.add_row([r['target'], r['attack'],
                         f"{r['plain_acc']*100:.2f}%", f"{r['plain_f1']*100:.2f}%",
                         f"{r['adv_acc']*100:.2f}%",   f"{r['adv_f1']*100:.2f}%",
                         f"{r['asr']*100:.2f}%"])
        return tbl

    @staticmethod
    def compact_table(results):
        """Build a compact 2D PrettyTable (rows=attacks, cols=targets).

        First row shows plain accuracy per target, subsequent rows show
        adv accuracy + ASR per attack×target cell.
        """
        from prettytable import PrettyTable
        targets = list(dict.fromkeys(r['target'] for r in results))
        attacks = list(dict.fromkeys(r['attack'] for r in results))

        tbl = PrettyTable(['Attack'] + targets)
        tbl.align = 'r'
        tbl.align['Attack'] = 'l'

        plain_row = ['original']
        for t in targets:
            v = next((r['plain_acc'] for r in results if r['target'] == t), float('nan'))
            plain_row.append(f"{v*100:.2f}%" if v == v else 'nan')
        tbl.add_row(plain_row)

        for atk in attacks:
            row = [atk]
            for t in targets:
                m = next((r for r in results if r['target'] == t and r['attack'] == atk), None)
                row.append(f"{m['adv_acc']*100:.2f}% (asr:{m['asr']*100:.2f}%)" if m else '—')
            tbl.add_row(row)
        return tbl

    @staticmethod
    def save_csv(results, path):
        """Save results list[dict] to CSV."""
        import pandas as pd
        pd.DataFrame(results).to_csv(path, index=False)
        logger.info(f"[+] Saved: {path}")


# ── Backward-compatible free functions ────────────────────────────────────────

def report_metrics(name, y_true, y_pred):
    """Log Accuracy, Macro-F1, Macro-TPR, Macro-FPR for a model."""
    return Evaluator().report(name, y_true, y_pred)


def final_predict(predictor, X, y_true=None):
    """Run predictor.predict(X) with NaN-sample handling."""
    return Evaluator.predict_safe(predictor, X, y_true)


def asr(y_plain_pred, y_adv_pred, y_true):
    """Attack Success Rate: fraction of originally-correct samples now misclassified."""
    return Evaluator.asr(y_plain_pred, y_adv_pred, y_true)
