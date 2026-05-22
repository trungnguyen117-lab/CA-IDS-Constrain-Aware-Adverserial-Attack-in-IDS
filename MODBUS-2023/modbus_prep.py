"""MODBUS-specific clean + split + balance — tái tạo notebook 0 & 1.

Logic không share với IEC-104 vì format raw + recipe split khác:
- Raw có space trong tên cột, có Src_Port/Dst_Port phải drop tay
- Label encode theo recipe BENIGN=0 first, các lớp khác sort alphabet
- Split: lớp abundant (0, 2) lấy cố định N test, phần còn lại vào train;
  lớp rare stratified 70/30

Hằng số recipe nằm dưới ``cfg.augment.modbus`` để không phải đụng
``src/core/config.py`` (shared module).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.core.config import Config

logger = logging.getLogger("modbus.prep")


@dataclass(frozen=True)
class ModbusRecipe:
    """Hằng số recipe từ notebook 0 + 1."""

    label_col: str
    drop_manual: tuple[str, ...]
    abundant_labels: tuple[int, ...]
    test_n_per_abundant: int
    balance_n_per_abundant: int
    rare_test_ratio: float
    seed: int

    @classmethod
    def from_config(cls, cfg: Config) -> "ModbusRecipe":
        m = cfg.augment.get("modbus", {})
        return cls(
            label_col=cfg.label_col,
            drop_manual=tuple(m.get("drop_manual", ["Src_Port", "Dst_Port"])),
            abundant_labels=tuple(m.get("abundant_labels", [0, 2])),
            test_n_per_abundant=int(m.get("test_n_per_abundant", 600)),
            balance_n_per_abundant=int(m.get("balance_n_per_abundant", 1400)),
            rare_test_ratio=float(m.get("rare_test_ratio", 0.3)),
            seed=cfg.split_seed,
        )


@dataclass(frozen=True)
class CleanReport:
    n_rows: int
    n_cols: int
    constant_dropped: list[str]
    manual_dropped: list[str]
    label_mapping: dict[str, int]


class ModbusCleaner:
    """Notebook 0: normalize cols → drop NaN/constant → numeric only → encode label."""

    def __init__(self, recipe: ModbusRecipe) -> None:
        self.r = recipe

    def clean(self, df_raw: pd.DataFrame) -> tuple[pd.DataFrame, CleanReport]:
        df = df_raw.copy()
        df.columns = [c.strip().replace(" ", "_") for c in df.columns]

        df = df.dropna(axis=1, how="all")

        label_col = self.r.label_col
        if label_col not in df.columns:
            raise KeyError(f"Missing label column {label_col!r}")

        constants = [c for c in df.columns
                     if c != label_col and df[c].nunique(dropna=False) <= 1]
        if constants:
            df = df.drop(columns=constants)
            logger.info("Dropped %d constant cols: %s", len(constants), constants)

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        keep = numeric_cols + ([label_col] if label_col not in numeric_cols else [])
        df = df[keep] if label_col in numeric_cols else df[numeric_cols + [label_col]]

        labels = df[label_col].astype(str).str.strip().str.upper()
        uniq = sorted(labels.unique())
        if "BENIGN" in uniq:
            uniq.remove("BENIGN")
            order = ["BENIGN"] + uniq
        else:
            order = uniq
        mapping = {lbl: i for i, lbl in enumerate(order)}
        df[label_col] = labels.map(mapping).astype(int)

        manual = [c for c in self.r.drop_manual if c in df.columns]
        if manual:
            df = df.drop(columns=manual)
            logger.info("Dropped manual cols: %s", manual)

        df = df.reset_index(drop=True)
        logger.info("Clean shape: %s", df.shape)
        return df, CleanReport(
            n_rows=int(df.shape[0]),
            n_cols=int(df.shape[1]),
            constant_dropped=constants,
            manual_dropped=manual,
            label_mapping=mapping,
        )


class ModbusSplitter:
    """Notebook 1 cell 2-3: per-class split + balance."""

    def __init__(self, recipe: ModbusRecipe) -> None:
        self.r = recipe

    def split(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Lớp abundant: lấy ``test_n_per_abundant`` mẫu test, còn lại → train.
        Lớp khác: stratified split theo ``rare_test_ratio``.
        """
        r = self.r
        label_col = r.label_col

        test_parts: list[pd.DataFrame] = []
        train_parts: list[pd.DataFrame] = []

        for lbl in r.abundant_labels:
            sub = df[df[label_col] == lbl]
            if len(sub) < r.test_n_per_abundant:
                raise ValueError(
                    f"Label {lbl}: {len(sub)} samples < test_n_per_abundant={r.test_n_per_abundant}"
                )
            test_part = sub.sample(n=r.test_n_per_abundant, random_state=r.seed)
            train_part = sub.drop(test_part.index)
            test_parts.append(test_part)
            train_parts.append(train_part)
            logger.info("  abundant label %s: total=%d → train=%d test=%d",
                        lbl, len(sub), len(train_part), len(test_part))

        others = df[~df[label_col].isin(r.abundant_labels)]
        if len(others):
            tr_o, te_o = train_test_split(
                others,
                test_size=r.rare_test_ratio,
                stratify=others[label_col],
                random_state=r.seed,
            )
            train_parts.append(tr_o)
            test_parts.append(te_o)
            logger.info("  rare classes: total=%d → train=%d test=%d (ratio=%g)",
                        len(others), len(tr_o), len(te_o), r.rare_test_ratio)

        train_df = pd.concat(train_parts, ignore_index=True)
        test_df = pd.concat(test_parts, ignore_index=True)
        return train_df, test_df

    def balance(self, train: pd.DataFrame) -> pd.DataFrame:
        """Cap mỗi lớp abundant về ``balance_n_per_abundant`` mẫu."""
        r = self.r
        label_col = r.label_col

        keep_idx: list = []
        rng = np.random.default_rng(r.seed)

        for lbl in r.abundant_labels:
            idx = train.index[train[label_col] == lbl].to_numpy()
            if len(idx) < r.balance_n_per_abundant:
                raise ValueError(
                    f"Label {lbl}: {len(idx)} < balance_n_per_abundant={r.balance_n_per_abundant}"
                )
            sampled = rng.choice(idx, size=r.balance_n_per_abundant, replace=False)
            keep_idx.extend(sampled.tolist())

        other_idx = train.index[~train[label_col].isin(r.abundant_labels)].tolist()
        keep_idx.extend(other_idx)

        balanced = train.loc[keep_idx].reset_index(drop=True)
        logger.info("Balanced shape: %s", balanced.shape)
        return balanced


def class_dist(df: pd.DataFrame, label_col: str) -> dict[int, int]:
    """Trả về dict {label: count} sắp xếp tăng dần theo label."""
    vc = df[label_col].value_counts().sort_index()
    return {int(k): int(v) for k, v in vc.items()}
