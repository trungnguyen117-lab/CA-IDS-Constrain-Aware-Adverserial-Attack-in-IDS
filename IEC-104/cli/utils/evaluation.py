"""Evaluation metrics: accuracy, F1, TPR, FPR, ASR, confusion matrix."""

import logging

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

logger = logging.getLogger(__name__)


def macro_tpr_fpr(cm):
    """Compute macro TPR and FPR from confusion matrix."""
    n = cm.shape[0]
    tpr_list, fpr_list = [], []
    for i in range(n):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = cm.sum() - tp - fn - fp
        tpr_list.append(tp / (tp + fn) if (tp + fn) > 0 else 0.0)
        fpr_list.append(fp / (fp + tn) if (fp + tn) > 0 else 0.0)
    return np.mean(tpr_list), np.mean(fpr_list)


def compute_asr(y_true, y_clean_pred, y_adv_pred):
    """Attack Success Rate on correctly-classified samples."""
    correct = np.where(y_true == y_clean_pred)[0]
    if len(correct) == 0:
        return 0.0
    return np.sum(y_clean_pred[correct] != y_adv_pred[correct]) / len(correct) * 100


def report_metrics(name, y_true, y_pred):
    """Log accuracy, macro-F1, macro-TPR, macro-FPR."""
    acc = accuracy_score(y_true, y_pred) * 100
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0) * 100
    cm = confusion_matrix(y_true, y_pred)
    tpr, fpr = macro_tpr_fpr(cm)
    logger.info(
        f"{name:>25s}  Acc={acc:6.2f}%  F1={f1:6.2f}%  TPR={tpr*100:6.2f}%  FPR={fpr*100:6.2f}%"
    )
    return {"acc": acc, "f1": f1, "tpr": tpr * 100, "fpr": fpr * 100}


def _fmt(val):
    """Format a numeric value or return '-'."""
    if val is None:
        return "-"
    return f"{val:.2f}%"


def _delta_cell(before, after, higher_is_better=True):
    """Format 'before → after (delta)' with Rich color markup."""
    if before is None and after is None:
        return "-"
    if before is None:
        return f"{after:.2f}%"
    if after is None:
        return f"{before:.2f}%"

    d = after - before
    sign = "+" if d >= 0 else ""
    improved = (d > 0) if higher_is_better else (d < 0)
    color = "green" if improved else ("red" if not improved and d != 0 else "dim")
    return f"{before:.1f} → {after:.1f} [{color}]({sign}{d:.1f})[/{color}]"


def _collect_attacks(all_results):
    """Collect unique attack names across all targets, preserving order."""
    attacks = []
    for results in all_results.values():
        for name in results:
            if name not in attacks:
                attacks.append(name)
    return attacks


def _build_rich_table(all_results, title):
    """Build a Rich Table: models as columns, metrics as rows."""
    from rich.table import Table

    targets = list(all_results.keys())
    all_attacks = _collect_attacks(all_results)

    t = Table(title=title, show_lines=True, title_style="bold cyan")
    t.add_column("Metric", style="bold", min_width=20)
    for tgt in targets:
        t.add_column(tgt.upper(), justify="right", min_width=10)

    for key, label in [("acc", "Acc (clean)"), ("f1", "F1 (clean)")]:
        row = [label]
        for tgt in targets:
            row.append(_fmt(all_results[tgt].get("clean", {}).get(key)))
        t.add_row(*row)

    t.add_section()

    for atk in all_attacks:
        if atk == "clean":
            continue
        row = [f"ASR ({atk})"]
        for tgt in targets:
            row.append(_fmt(all_results[tgt].get(atk, {}).get("asr")))
        t.add_row(*row)

    t.add_section()

    for atk in all_attacks:
        if atk == "clean":
            continue
        row = [f"Acc ({atk})"]
        for tgt in targets:
            row.append(_fmt(all_results[tgt].get(atk, {}).get("acc")))
        t.add_row(*row)

    return t


def _build_comparison_table(baseline, at):
    """Build Rich Table comparing before/after AT with colored deltas."""
    from rich.table import Table

    targets = list(baseline.keys())
    all_attacks = _collect_attacks(baseline)

    t = Table(title="BEFORE vs AFTER AT", show_lines=True, title_style="bold cyan")
    t.add_column("Metric", style="bold", min_width=20)
    for tgt in targets:
        t.add_column(tgt.upper(), justify="right", min_width=22)

    # Acc/F1 clean — higher is better
    for key, label in [("acc", "Acc (clean)"), ("f1", "F1 (clean)")]:
        row = [label]
        for tgt in targets:
            b = baseline[tgt].get("clean", {}).get(key)
            a = at.get(tgt, {}).get("clean", {}).get(key)
            row.append(_delta_cell(b, a, higher_is_better=True))
        t.add_row(*row)

    t.add_section()

    # ASR — lower is better
    for atk in all_attacks:
        if atk == "clean":
            continue
        row = [f"ASR ({atk})"]
        for tgt in targets:
            b = baseline[tgt].get(atk, {}).get("asr")
            a = at.get(tgt, {}).get(atk, {}).get("asr")
            row.append(_delta_cell(b, a, higher_is_better=False))
        t.add_row(*row)

    t.add_section()

    # Acc under attack — higher is better
    for atk in all_attacks:
        if atk == "clean":
            continue
        row = [f"Acc ({atk})"]
        for tgt in targets:
            b = baseline[tgt].get(atk, {}).get("acc")
            a = at.get(tgt, {}).get(atk, {}).get("acc")
            row.append(_delta_cell(b, a, higher_is_better=True))
        t.add_row(*row)

    return t


def print_summary(all_results, title="SUMMARY"):
    """Print summary table using Rich."""
    from rich.console import Console
    Console().print(_build_rich_table(all_results, title))


def print_comparison_all(baseline, at):
    """Print combined before/after AT comparison with colored deltas."""
    from rich.console import Console
    Console().print(_build_comparison_table(baseline, at))
