"""Data loading and metadata management for foami+ pipeline scripts."""
import os
import logging

import numpy as np
import pandas as pd

from .constants import (
    LABEL_COL, GBT_TARGETS,
    BLACKBOX_ATTACKS, WHITEBOX_ATTACKS, DL_FALLBACK_TARGET,
)
from .loaders import require_file
from .paths import adv_csv

logger = logging.getLogger(__name__)


# ── Standalone augmentation / merge functions ─────────────────────────────────

def gaussian_augment(X, y, sigma, ratio, clip_range=None, mode='augment'):
    """Generate noisy copies of input data.

    Args:
        X: feature array (n_samples, n_features)
        y: label array (n_samples,)
        sigma: std deviation of Gaussian noise
        ratio: number of noisy copies per original sample (ignored in replace mode)
        clip_range: (min, max) tuple to clip values, or None
        mode: 'augment' = keep original + add noisy copies
              'replace' = replace original with single noisy version

    Returns:
        X_out, y_out
    """
    if mode == 'replace':
        noise = np.random.normal(0, sigma, size=X.shape).astype(X.dtype)
        X_out = X + noise
        if clip_range is not None:
            X_out = np.clip(X_out, clip_range[0], clip_range[1])
        return X_out, y

    parts_X = [X]
    parts_y = [y]
    for _ in range(ratio):
        noise = np.random.normal(0, sigma, size=X.shape).astype(X.dtype)
        X_noisy = X + noise
        if clip_range is not None:
            X_noisy = np.clip(X_noisy, clip_range[0], clip_range[1])
        parts_X.append(X_noisy)
        parts_y.append(y)

    return np.concatenate(parts_X, axis=0), np.concatenate(parts_y, axis=0)


def tvae_augment(
    df_train: pd.DataFrame,
    label_col: str,
    sample_max: int,
    labels: list,
    use_cuda: bool,
) -> pd.DataFrame:
    """Augment df_train with TVAE-generated synthetic samples.

    For each label in `labels` whose count < sample_max, fits a
    TVAESynthesizer on that class subset and samples the missing rows.

    Parameters
    ----------
    df_train   : original training DataFrame (features + label_col)
    label_col  : name of the label column
    sample_max : target sample count per class
    labels     : class values to augment (only those below sample_max)
    use_cuda   : whether to use CUDA for TVAE training

    Returns
    -------
    DataFrame with original rows + synthetic rows, duplicates removed.
    """
    try:
        from sdv.single_table import TVAESynthesizer
        from sdv.metadata import Metadata
    except ImportError:
        raise SystemExit(
            "SDV not installed. Install with: pip install sdv"
        )

    synthetic_dfs = []

    for label in labels:
        df_label = df_train[df_train[label_col] == label]
        current  = len(df_label)
        needed   = sample_max - current

        if needed <= 0:
            logger.info(f"  Label {label}: {current} samples — no augmentation needed")
            continue

        logger.info(f"  Label {label}: {current} samples → generating {needed} synthetic")

        metadata    = Metadata.detect_from_dataframe(
            data=df_label, table_name=str(label)
        )
        synthesizer = TVAESynthesizer(
            metadata,
            embedding_dim=64,
            compress_dims=[128, 64],
            decompress_dims=[64, 128],
            l2scale=1e-4,
            loss_factor=2.0,
            batch_size=512,
            epochs=256,
            cuda=use_cuda,
        )
        synthesizer.fit(df_label)
        df_synth             = synthesizer.sample(num_rows=needed)
        df_synth[label_col]  = label   # ensure label column is correct
        synthetic_dfs.append(df_synth)

    if not synthetic_dfs:
        logger.info("[+] All classes already at or above sample_max — returning original data")
        return df_train.copy()

    augmented = pd.concat([df_train] + synthetic_dfs, axis=0, ignore_index=True)

    # Remove duplicates (feature columns only, keep original)
    before     = len(augmented)
    feat_cols  = [c for c in augmented.columns if c != label_col]
    dup_mask   = augmented[feat_cols].duplicated(keep='first')
    augmented  = augmented[~dup_mask].reset_index(drop=True)
    removed    = before - len(augmented)

    if removed:
        logger.info(f"[+] Removed {removed} duplicate rows")

    return augmented


def merge_adv_csvs(
    base_csv: str,
    targets: list,
    attacks: list,
    adv_dir: str,
    drop_duplicates: bool,
    extra_csvs: list = None,
) -> pd.DataFrame:
    """Load base CSV + optional extra CSVs + adversarial CSVs, merge, optionally dedup.

    Args:
        base_csv:       Primary base training CSV (e.g. TVAE-augmented train_at.csv).
        targets:        Model target names whose adv CSVs to include.
        attacks:        Attack names whose adv CSVs to include.
        adv_dir:        Root directory for adversarial CSVs.
        drop_duplicates: Remove duplicate feature rows after merge.
        extra_csvs:     Additional CSVs to include before adversarial data
                        (e.g. original train_shap_66.csv).
    """

    if not os.path.exists(base_csv):
        raise SystemExit(
            f"Base training CSV not found: {base_csv}\n"
            "Run prepare_adv_data.py first."
        )

    logger.info(f"[+] Loading base data: {base_csv}")
    df_base   = pd.read_csv(base_csv, low_memory=False)
    label_col = LABEL_COL
    feat_cols = [c for c in df_base.columns if c != label_col]
    logger.info(f"[+] Base shape: {df_base.shape}")

    parts = [df_base]
    loaded, skipped = 0, 0

    for extra in (extra_csvs or []):
        if not os.path.exists(extra):
            logger.warning(f"  [!] Extra CSV not found, skip: {extra}")
            continue
        df_extra = pd.read_csv(extra, low_memory=False)
        extra_fc = [c for c in df_extra.columns if c != label_col]
        if extra_fc != feat_cols:
            logger.warning(f"  [!] Extra CSV feature mismatch ({len(extra_fc)} vs {len(feat_cols)}), skip: {extra}")
            continue
        parts.append(df_extra)
        logger.info(f"  + extra: {os.path.basename(extra)}: {df_extra.shape[0]:>7,} rows")

    for target in targets:
        for attack in attacks:
            path = adv_csv(target, attack, adv_dir)
            if not os.path.exists(path):
                logger.debug(f"  skip (not found): {path}")
                skipped += 1
                continue

            df_adv = pd.read_csv(path, low_memory=False)

            # Align columns to base (in case adv CSV has extra/missing cols)
            adv_cols = [c for c in df_adv.columns if c != label_col]
            if adv_cols != feat_cols:
                logger.warning(
                    f"  {target}/{attack}: feature mismatch "
                    f"({len(adv_cols)} vs {len(feat_cols)}) — skip"
                )
                skipped += 1
                continue

            parts.append(df_adv)
            logger.info(f"  + {target:8s} / {attack:8s}: {df_adv.shape[0]:>7,} rows")
            loaded += 1

    logger.info(f"[+] Loaded {loaded} adversarial CSV(s), skipped {skipped}")

    merged = pd.concat(parts, axis=0, ignore_index=True)
    logger.info(f"[+] Merged shape (before dedup): {merged.shape}")

    if drop_duplicates:
        before  = len(merged)
        dup     = merged[feat_cols].duplicated(keep='first')
        merged  = merged[~dup].reset_index(drop=True)
        removed = before - len(merged)
        if removed:
            logger.info(f"[+] Removed {removed:,} duplicate rows")

    # ── Class distribution ─────────────────────────────────────────────────
    dist = merged[label_col].value_counts().sort_index()
    logger.info(f"[+] Class distribution:\n{dist.to_string()}")
    logger.info(f"[+] Final shape: {merged.shape}")

    return merged


def _build_adv_pairs(model: str, attacks: list = None) -> list[tuple[str, str]]:
    """Build (target, attack) pairs for per-model adversarial training.

    Tree models (xgb, cat, rf):
      - own blackbox attacks: {model}_zoo, {model}_hsja
      - DL fallback whitebox: lstm_deepfool, lstm_fgsm, lstm_cw, lstm_pgd, lstm_jsma

    DL models (lstm, resdnn):
      - all own attacks: {model}_zoo, {model}_hsja, {model}_deepfool, ...
    """
    if model in GBT_TARGETS:
        bb = [(model, a) for a in BLACKBOX_ATTACKS]
        wb = [(DL_FALLBACK_TARGET, a) for a in WHITEBOX_ATTACKS]
        pairs = bb + wb
    else:
        pairs = [(model, a) for a in (attacks or BLACKBOX_ATTACKS + WHITEBOX_ATTACKS)]

    if attacks:
        allowed = set(attacks)
        pairs = [(t, a) for t, a in pairs if a in allowed]

    return pairs


def _balanced_sample(df: pd.DataFrame, n: int, label_col: str = LABEL_COL) -> pd.DataFrame:
    """Sample up to n rows, balanced per label (stratified)."""
    labels = df[label_col].unique()
    per_label = max(1, n // len(labels))
    parts = []
    for lbl in labels:
        subset = df[df[label_col] == lbl]
        take = min(len(subset), per_label)
        parts.append(subset.sample(n=take, random_state=42))
    return pd.concat(parts, ignore_index=True)


def merge_per_model(
    model: str,
    base_csv: str,
    adv_dir: str,
    attacks: list = None,
    sample_threshold: int = -1,
    drop_duplicates: bool = True,
    extra_csvs: list = None,
) -> pd.DataFrame:
    """Merge base CSV + adversarial CSVs for a single model's AT training.

    Args:
        model:            Target model name (xgb, cat, rf, lstm, resdnn).
        base_csv:         Base training CSV (e.g. train_tvae.csv).
        adv_dir:          Root adversarial samples directory.
        attacks:          Filter to specific attacks (default: all compatible).
        sample_threshold: Max adv samples per adv CSV (-1 = no limit), balanced per label.
        drop_duplicates:  Remove duplicate feature rows after merge.
        extra_csvs:       Additional CSVs to include before adversarial data.
    """
    if not os.path.exists(base_csv):
        raise SystemExit(f"Base training CSV not found: {base_csv}")

    logger.info(f"[+] merge_per_model({model})")
    logger.info(f"  base: {base_csv}")

    df_base = pd.read_csv(base_csv, low_memory=False)
    feat_cols = [c for c in df_base.columns if c != LABEL_COL]
    logger.info(f"  base shape: {df_base.shape}")

    parts = [df_base]

    for extra in (extra_csvs or []):
        if not os.path.exists(extra):
            logger.warning(f"  [!] Extra CSV not found, skip: {extra}")
            continue
        df_extra = pd.read_csv(extra, low_memory=False)
        extra_fc = [c for c in df_extra.columns if c != LABEL_COL]
        if extra_fc != feat_cols:
            logger.warning(f"  [!] Extra CSV feature mismatch, skip: {extra}")
            continue
        parts.append(df_extra)
        logger.info(f"  + extra: {os.path.basename(extra)}: {df_extra.shape[0]:>7,} rows")

    pairs = _build_adv_pairs(model, attacks)
    logger.info(f"  adv pairs: {pairs}")

    adv_parts = []
    for target, attack in pairs:
        path = adv_csv(target, attack, adv_dir)
        if not os.path.exists(path):
            logger.debug(f"  skip (not found): {path}")
            continue
        df_adv = pd.read_csv(path, low_memory=False)
        adv_cols = [c for c in df_adv.columns if c != LABEL_COL]
        if adv_cols != feat_cols:
            logger.warning(f"  {target}/{attack}: feature mismatch — skip")
            continue
        adv_parts.append(df_adv)
        logger.info(f"  + {target:8s} / {attack:8s}: {df_adv.shape[0]:>7,} rows")

    if sample_threshold > 0 and adv_parts:
        adv_parts = [_balanced_sample(df, sample_threshold) for df in adv_parts]
        total_adv = sum(len(df) for df in adv_parts)
        logger.info(f"  [threshold] max {sample_threshold} per adv CSV → {total_adv} total adv samples")

    parts.extend(adv_parts)
    merged = pd.concat(parts, axis=0, ignore_index=True)
    logger.info(f"  merged shape (before dedup): {merged.shape}")

    if drop_duplicates:
        before = len(merged)
        dup = merged[feat_cols].duplicated(keep='first')
        merged = merged[~dup].reset_index(drop=True)
        removed = before - len(merged)
        if removed:
            logger.info(f"  removed {removed:,} duplicate rows")

    dist = merged[LABEL_COL].value_counts().sort_index()
    logger.info(f"  class distribution:\n{dist.to_string()}")
    logger.info(f"  final shape: {merged.shape}")

    return merged


class DataManager:
    """Manages dataset loading, splitting, and metadata.

    Usage:
        dm = DataManager(train_csv, test_csv)
        X_train, y_train = dm.train_data
        X_test, y_test = dm.test_data
        print(dm.num_classes, dm.input_dim, dm.clip_values)

        # Optional val split
        X_train, y_train, X_val, y_val = dm.split_val(val_frac=0.1)
    """

    def __init__(self, train_csv=None, test_csv=None, label_col=LABEL_COL):
        self.label_col = label_col
        self._train = None
        self._test = None
        if train_csv:
            self._train = self._load(train_csv)
        if test_csv:
            self._test = self._load(test_csv)

    def _load(self, path):
        require_file(path)
        df = pd.read_csv(path, low_memory=False)
        feat_cols = [c for c in df.columns if c != self.label_col]
        X = df[feat_cols].values.astype(np.float32)
        y = df[self.label_col].values.astype(np.int64)
        return X, y

    @property
    def train_data(self):
        return self._train

    @property
    def test_data(self):
        return self._test

    def _any_data(self):
        """Return first available dataset (train preferred)."""
        data = self._train or self._test
        if data is None:
            raise ValueError("No data loaded — provide train_csv or test_csv")
        return data

    @property
    def num_classes(self):
        X, y = self._any_data()
        return int(len(np.unique(y)))

    @property
    def input_dim(self):
        X, y = self._any_data()
        return X.shape[1]

    @property
    def clip_values(self):
        X, y = self._any_data()
        return (float(X.min()), float(X.max()))

    def split_val(self, val_frac=0.1):
        """Split training data into train + val. Returns (X_t, y_t, X_v, y_v)."""
        X, y = self._train
        n = len(X)
        n_val = max(1, int(n * val_frac))
        idx = np.random.permutation(n)
        return X[idx[n_val:]], y[idx[n_val:]], X[idx[:n_val]], y[idx[:n_val]]
