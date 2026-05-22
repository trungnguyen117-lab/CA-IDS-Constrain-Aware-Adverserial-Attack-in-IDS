"""Metrics + AT data assembly + adv CSV I/O."""

from __future__ import annotations

import glob
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, precision_score, recall_score,
)


# ── evaluation ──

logger = logging.getLogger(__name__)


def report_metrics(name, y_true, y_pred, labels=None) -> dict:
    kw = dict(average="macro", zero_division=0)
    acc = accuracy_score(y_true, y_pred) * 100
    f1 = f1_score(y_true, y_pred, **kw) * 100
    prec = precision_score(y_true, y_pred, **kw) * 100
    dr = recall_score(y_true, y_pred, **kw) * 100   # macro DR = macro TPR
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    logger.info("%-25s  Acc=%6.2f%%  F1=%6.2f%%  P=%6.2f%%  DR=%6.2f%%",
                name, acc, f1, prec, dr)
    return {
        "acc": acc, "macro_f1": f1,
        "macro_prec": prec, "macro_dr": dr,
        "confusion_matrix": cm.tolist(),
    }


def compute_asr(y_true, y_clean, y_adv) -> float:
    """% flipped on samples model originally predicted correctly."""
    correct = y_clean == y_true
    n_correct = int(correct.sum())
    if n_correct == 0:
        return 0.0
    flipped = (y_adv[correct] != y_true[correct]).sum()
    return 100.0 * float(flipped) / n_correct


def load_adv_features(path, feature_names, expected_n: int | None = None):
    """Load adversarial CSV features, optionally skipping row-count mismatches."""
    if not os.path.isfile(path):
        logger.warning("Missing: %s", path)
        return None
    df = pd.read_csv(path)
    if expected_n is not None and len(df) != expected_n:
        logger.warning("Skip %s: row mismatch (%d vs %d)", path, len(df), expected_n)
        return None
    missing = [c for c in feature_names if c not in df.columns]
    if missing:
        logger.warning(
            "Skip %s: feature schema mismatch (%d missing): %s",
            path, len(missing), ", ".join(missing),
        )
        return None
    return df[feature_names].values.astype(np.float32)


def _scan_dir(base_dir: Path) -> dict[str, str]:
    out = {}
    if not base_dir.is_dir():
        return out
    for path in sorted(glob.glob(str(base_dir / "*_adv.csv"))):
        stem = Path(path).name[:-len("_adv.csv")]
        if "_" in stem:
            out[stem.rsplit("_", 1)[-1]] = path
    return out


def find_adv_csvs(target: str, base_dir, cfg=None) -> dict[str, str]:
    """{attack: path} for ``{target}_{attack}_adv.csv`` under base_dir/target/.

    If ``cfg`` is given and ``target`` is a tree, also scan WB attacks from
    ``cfg.transfer_sources[target]`` (e.g. tree → mlp/ftt/resdnn). Own files
    take priority; transfer source fills only the gaps.
    """
    base_dir = Path(base_dir)
    out = _scan_dir(base_dir / target)
    if cfg is not None:
        allowed = set(cfg.wb_attacks) | set(cfg.bb_attacks)
        out = {a: p for a, p in out.items() if a in allowed}

    if cfg is not None and target not in cfg.surrogate_targets:
        for src in cfg.transfer_sources.get(target, []):
            for atk, path in _scan_dir(base_dir / src).items():
                if atk in cfg.wb_attacks and atk not in out:
                    out[atk] = path
    return out


def discover_attacks(adv_dir) -> list[str]:
    """List attack names appearing under ``adv_dir`` (any target)."""
    base = Path(adv_dir)
    if not base.is_dir():
        return []
    attacks: set[str] = set()
    for sub in base.iterdir():
        if not sub.is_dir():
            continue
        for f in sub.iterdir():
            if f.name.endswith("_adv.csv"):
                stem = f.name[:-len("_adv.csv")]
                if "_" in stem:
                    attacks.add(stem.rsplit("_", 1)[-1])
    return sorted(attacks)


# ── masking ──

def get_mutate_indices(df: pd.DataFrame, label_col: str = "Label",
                       cont_features: list[str] | None = None,
                       extra_protected: list[str] | None = None) -> list[int]:
    """Index of columns to PROTECT (mask=0).

    - If ``cont_features`` is given: protect every column NOT in the list
      (single source of truth from cfg.cont_features).
    - Otherwise: protect binary {0,1} columns + ``extra_protected`` names.
    """
    feature_cols = [c for c in df.columns if c != label_col]
    if cont_features:
        cont_set = set(cont_features)
        return [i for i, col in enumerate(feature_cols) if col not in cont_set]
    out = []
    for i, col in enumerate(feature_cols):
        if extra_protected and col in extra_protected:
            out.append(i)
            continue
        vals = df[col].dropna()
        if vals.nunique() == 2 and set(vals.unique()).issubset({0, 1, 0.0, 1.0}):
            out.append(i)
    return out


# ── at_assembly ──

logger = logging.getLogger(__name__)


def hash_rows(X: np.ndarray, feat_names) -> set:
    df = pd.DataFrame(np.round(X.astype(np.float64), 5), columns=feat_names)
    return set(pd.util.hash_pandas_object(df, index=False).values)


def load_filter_adv(path, clean_hashes, feat_names, label_col="Label"):
    if not os.path.isfile(path):
        return None
    df = pd.read_csv(path)
    if not set(feat_names).issubset(df.columns):
        return None
    h_adv = pd.util.hash_pandas_object(
        df[feat_names].astype(np.float64).round(5), index=False
    ).values
    mask = np.array([h not in clean_hashes for h in h_adv])
    n_drop = int((~mask).sum())
    df = df[mask].reset_index(drop=True)
    logger.info("  %s: %d → %d (drop %d match clean)",
                os.path.basename(path), len(df) + n_drop, len(df), n_drop)
    return df if len(df) > 0 else None


def list_adv_csvs(target_dir, target_name, attack_filter=None):
    pat = os.path.join(target_dir, f"{target_name}_*_train_adv.csv")
    paths = sorted(glob.glob(pat))
    if attack_filter:
        keep = set(attack_filter)
        paths = [p for p in paths
                 if os.path.basename(p)[len(target_name) + 1:-len("_train_adv.csv")] in keep]
    return paths


def collect_attack_dfs(cfg, target, clean_hashes, feat_names, label_col,
                       adv_base, attack_filter):
    """Collect direct + transfer adv dfs grouped by attack name.

    Returns ``{attack: [dfs...]}``. Skips dfs that are empty after filtering
    out rows duplicating the clean set.
    """
    def adv_dir(t):
        return (os.path.join(adv_base, "adv_training", t) if adv_base
                else str(cfg.adv_train_path(t)))

    out: dict[str, list[pd.DataFrame]] = {}

    def add(atk, df):
        if df is not None:
            out.setdefault(atk, []).append(df)

    # direct adv
    for path in list_adv_csvs(adv_dir(target), target, attack_filter):
        atk = os.path.basename(path)[len(target) + 1:-len("_train_adv.csv")]
        add(atk, load_filter_adv(path, clean_hashes, feat_names, label_col))

    # Transfer adv for tree targets: WB attacks only.
    # Trees already have their own BB adv (zoo/hsja) directly — pulling DNN's
    # BB transfer is redundant and noisy (different model, different boundary).
    if cfg.is_tree(target):
        wb = set(cfg.wb_attacks)
        transfer_filter = (set(attack_filter) & wb) if attack_filter else wb
        for src in cfg.transfer_sources.get(target, []):
            for path in list_adv_csvs(adv_dir(src), src, transfer_filter):
                atk = os.path.basename(path)[len(src) + 1:-len("_train_adv.csv")]
                add(atk, load_filter_adv(path, clean_hashes, feat_names, label_col))

    return out


def balance_per_class(attack_dfs, clean_df, label_col, *,
                      attack_weights, per_attack_cap, adv_clean_ratio, seed=42):
    """Class-balanced sampling across attacks. Returns ``(adv_df, contrib, avail)``.

    ``contrib[attack][class]`` = rows kept; ``avail[attack][class]`` = rows had.
    """
    rng = np.random.RandomState(seed)
    clean_per_class = clean_df[label_col].value_counts().to_dict()
    weights = {atk: 1.0 for atk in attack_dfs}
    if attack_weights:
        for atk, w in attack_weights.items():
            if atk in weights:
                weights[atk] = float(w)
    wsum = sum(weights.values())
    all_classes = sorted({l for df in attack_dfs.values() for l in df[label_col].unique()})
    n_classes = len(all_classes)
    per_attack_class_cap = (
        int(np.ceil(per_attack_cap / max(n_classes, 1)))
        if per_attack_cap is not None else None
    )

    parts: list[pd.DataFrame] = []
    contrib: dict = {atk: {} for atk in attack_dfs}
    avail: dict = {atk: {} for atk in attack_dfs}
    for c in all_classes:
        budget = int(clean_per_class.get(c, 0)) * float(adv_clean_ratio)
        for atk, df in attack_dfs.items():
            sub = df[df[label_col] == c]
            avail[atk][c] = len(sub)
            quota = int(round(budget * weights[atk] / wsum))
            if per_attack_class_cap is not None:
                quota = min(quota, per_attack_class_cap)
            target_n = min(len(sub), quota)
            contrib[atk][c] = target_n
            if target_n > 0:
                if len(sub) > target_n:
                    sub = sub.sample(n=target_n, random_state=rng)
                parts.append(sub)

    adv_df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    return adv_df, contrib, avail, weights, all_classes


def log_breakdown(attack_dfs, contrib, avail, all_classes, cfg, weights,
                  adv_clean_ratio, n_kept):
    """Pretty per-attack × per-class table to logger.info."""
    cls_label = (lambda c: cfg.label_names[c] if 0 <= c < len(cfg.label_names) else str(c))
    logger.info("Adv per-source breakdown (taken / available):")
    header = "  {:<14s} {:>8s}".format("attack", "total") + "".join(
        "  {:>14s}".format(cls_label(c)) for c in all_classes
    )
    logger.info(header)
    for atk in sorted(attack_dfs):
        tt = sum(contrib[atk].values())
        ta = sum(avail[atk].values())
        row = "  {:<14s} {:>8s}".format(atk, f"{tt}/{ta}") + "".join(
            "  {:>14s}".format(f"{contrib[atk].get(c, 0)}/{avail[atk].get(c, 0)}")
            for c in all_classes
        )
        logger.info(row)
    logger.info("Adv balanced → %d (weights=%s, ratio=%.2f)",
                n_kept, weights, adv_clean_ratio)


def assemble_at_data(cfg, target, X_clean, y_clean, feat_names,
                     adv_base=None, balance_adv=True, attack_filter=None,
                     attack_weights=None, per_attack_cap=None,
                     adv_clean_ratio=1.0):
    """Build (clean + adv) DataFrame for AT.

    cfg: Config — drives transfer rules and label_col.
    Tree targets: pull adv from ``transfer_sources[target]`` (e.g. MLP) too.
    DL targets: optionally pull HSJA adv from each tree target.
    """
    label_col = cfg.label_col
    clean_hashes = hash_rows(X_clean, feat_names)
    grouped = collect_attack_dfs(
        cfg, target, clean_hashes, feat_names, label_col, adv_base, attack_filter,
    )

    clean_df = pd.DataFrame(X_clean, columns=feat_names)
    clean_df[label_col] = y_clean

    if not grouped:
        logger.warning("No adv samples found — AT will == baseline")
        return clean_df

    attack_dfs = {atk: pd.concat(parts, ignore_index=True).drop_duplicates()
                  for atk, parts in grouped.items()}
    for atk, df in attack_dfs.items():
        logger.info("  Attack '%s': %d unique adv samples", atk, len(df))

    if balance_adv:
        adv_df, contrib, avail, weights, all_classes = balance_per_class(
            attack_dfs, clean_df, label_col,
            attack_weights=attack_weights,
            per_attack_cap=per_attack_cap,
            adv_clean_ratio=adv_clean_ratio,
        )
        log_breakdown(attack_dfs, contrib, avail, all_classes, cfg, weights,
                      adv_clean_ratio, len(adv_df))
    else:
        adv_df = pd.concat(attack_dfs.values(), ignore_index=True)

    out = pd.concat([clean_df, adv_df[feat_names + [label_col]]], ignore_index=True)
    pct = 100.0 * len(adv_df) / len(out)
    logger.info("AT data: clean=%d + adv=%d = %d (%.1f%% adv)",
                len(clean_df), len(adv_df), len(out), pct)
    return out


