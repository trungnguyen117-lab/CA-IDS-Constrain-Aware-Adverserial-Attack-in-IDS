"""Dataset preprocessing: augmentation (TVAE) + preparation (clean/leak-aware split)."""

from __future__ import annotations

import logging
import warnings
import pandas as pd
from .config import Config
import numpy as np
from sklearn.preprocessing import LabelEncoder
from .covas import CovasThresholds, dead_features, pair_feature_stats


# ── augment ──

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


class DataAugmentation:
    def __init__(self, cfg: Config):
        self.cfg = cfg

    # ── Single TVAE fit/sample core ───────────────────────────────────────
    def fit_tvae(self, df_class: pd.DataFrame, n: int) -> pd.DataFrame:
        """Fit a TVAE on ``df_class`` and sample ``n`` rows (Label re-attached)."""
        from sdv.metadata import Metadata
        from sdv.single_table import TVAESynthesizer

        cfg = self.cfg.augment.get("tvae", {})
        label_col = self.cfg.label_col
        lbl = df_class[label_col].iloc[0]

        meta = Metadata.detect_from_dataframe(
            data=df_class, table_name=f"class_{lbl}",
        )
        synth = TVAESynthesizer(
            meta,
            enforce_min_max_values=True,
            enforce_rounding=True,
            embedding_dim=cfg.get("embedding_dim", 64),
            compress_dims=tuple(cfg.get("compress_dims", (128, 64))),
            decompress_dims=tuple(cfg.get("decompress_dims", (64, 128))),
            l2scale=cfg.get("l2scale", 1e-4),
            loss_factor=cfg.get("loss_factor", 2.0),
            batch_size=cfg.get("batch_size", 512),
            epochs=cfg.get("epochs", 200),
            cuda=cfg.get("cuda", False),
        )
        synth.fit(df_class)
        out = synth.sample(num_rows=n)
        out[label_col] = lbl
        return out

    # ── Policy: balance every class to ``target`` ─────────────────────────
    def balance_to(
        self, df: pd.DataFrame, target: int | None = None,
    ) -> tuple[pd.DataFrame, dict]:
        """Subsample if ``n ≥ target``, TVAE top-up if ``n < target``.

        Returns ``(df_balanced, synth_counts_per_class)``.
        """
        cfg = self.cfg
        label_col = cfg.label_col
        seed = cfg.split_seed
        if target is None:
            target = cfg.augment.get("train_target", 8000)

        parts, synth_counts = [], {}
        for lbl in sorted(df[label_col].unique()):
            grp = df[df[label_col] == lbl].reset_index(drop=True)
            n = len(grp)
            if n >= target:
                kept = grp.sample(n=target, random_state=seed).reset_index(drop=True)
                synth_counts[int(lbl)] = 0
                logger.info("  class %s: %7d ≥ %d → subsample", lbl, n, target)
                parts.append(kept)
            else:
                n_gen = target - n
                logger.info("  class %s: %7d < %d → TVAE +%d", lbl, n, target, n_gen)
                syn = self.fit_tvae(grp, n_gen)
                synth_counts[int(lbl)] = int(len(syn))
                parts.append(pd.concat([grp, syn], ignore_index=True))

        out = pd.concat(parts, ignore_index=True).sample(
            frac=1.0, random_state=seed,
        ).reset_index(drop=True)
        return out, synth_counts

    # ── Policy: top-up only rare classes ──────────────────────────────────
    def topup_rare(
        self,
        df: pd.DataFrame,
        target_rare: int,
        rare: tuple[int, ...] | None = None,
    ) -> tuple[pd.DataFrame, dict]:
        """Keep abundant classes untouched; TVAE top-up classes in ``rare``
        until each reaches ``target_rare`` samples.
        """
        cfg = self.cfg
        label_col = cfg.label_col
        seed = cfg.split_seed
        if rare is None:
            rare = tuple(cfg.augment.get("rare_labels", (0, 1)))
        rare_set = {int(x) for x in rare}

        parts, synth_counts = [], {}
        for lbl, grp in df.groupby(label_col):
            grp = grp.reset_index(drop=True)
            n = len(grp)
            if int(lbl) in rare_set and n < target_rare:
                n_gen = target_rare - n
                logger.info("  class %s (rare) : %5d real → TVAE +%d", lbl, n, n_gen)
                syn = self.fit_tvae(grp, n_gen)
                parts.append(pd.concat([grp, syn], ignore_index=True))
                synth_counts[int(lbl)] = int(n_gen)
            else:
                logger.info("  class %s (keep) : %5d real (no TVAE)", lbl, n)
                parts.append(grp)
                synth_counts[int(lbl)] = 0

        out = pd.concat(parts, ignore_index=True).sample(
            frac=1.0, random_state=seed,
        ).reset_index(drop=True)
        return out, synth_counts


# ── prepare ──

logger = logging.getLogger(__name__)


class PrepareData:
    """Stateless transforms parameterised entirely by ``cfg``."""

    def __init__(self, cfg: Config):
        self.cfg = cfg

    # ── Stage 1: clean raw ────────────────────────────────────────────────
    def clean(self, df_raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
        """Drop non-numeric ID cols, encode label, replace NaN/Inf → 0.

        Returns ``(df_clean, label_mapping)``. Mapping is empty for binary mode.
        """
        cfg = self.cfg
        label_block = cfg.label
        mode = label_block.get("mode", "traffic")
        binary_col = label_block.get("binary_col", "Target")
        multi_col = label_block.get("multi_col", "Traffic")
        label_col = cfg.label_col

        df = df_raw.copy()
        existing_drop = [c for c in cfg.drop_cols if c in df.columns]
        df = df.drop(columns=existing_drop)
        logger.info("Dropped non-numeric ID cols: %s", existing_drop)

        mapping: dict = {}
        if mode == "traffic":
            if multi_col not in df.columns:
                raise KeyError(f"Missing label column {multi_col!r}")
            le = LabelEncoder()
            df[label_col] = le.fit_transform(df[multi_col].astype(str))
            mapping = {int(i): str(c) for i, c in enumerate(le.classes_)}
            df = df.drop(columns=[multi_col, binary_col], errors="ignore")
            logger.info("Traffic label mapping: %s", mapping)
        elif mode == "binary":
            if binary_col not in df.columns:
                raise KeyError(f"Missing label column {binary_col!r}")
            df[label_col] = df[binary_col].astype(int)
            df = df.drop(columns=[binary_col, multi_col], errors="ignore")
        else:
            raise ValueError(f"Unknown label mode: {mode}")

        feat_cols = [c for c in df.columns if c != label_col]
        non_numeric = [c for c in feat_cols
                       if not np.issubdtype(df[c].dtype, np.number)]
        if non_numeric:
            df = pd.get_dummies(df, columns=non_numeric, dtype=int)
            logger.info("One-hot encoded char cols %s → %d new binary cols",
                        non_numeric, df.shape[1] - len(feat_cols) - 1)

        feat_cols = [c for c in df.columns if c != label_col]
        n_nan = int(df[feat_cols].isna().sum().sum())
        n_inf = int(np.isinf(
            df[feat_cols].to_numpy(dtype=np.float64, na_value=0.0),
        ).sum())
        if n_nan or n_inf:
            df[feat_cols] = df[feat_cols].replace([np.inf, -np.inf], 0).fillna(0)
            logger.warning("Replaced NaN=%d, Inf=%d → 0", n_nan, n_inf)

        logger.info("Clean shape: %s  (features=%d)", df.shape, len(feat_cols))
        return df, mapping

    def drop_constants(self, df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
        """Drop feature columns with ≤ 1 unique value. Returns (df, dropped_names)."""
        label_col = self.cfg.label_col
        feats = [c for c in df.columns if c != label_col]
        constants = [c for c in feats if df[c].nunique(dropna=False) <= 1]
        if constants:
            df = df.drop(columns=constants)
        return df, constants

    def drop_dup_features(self, df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
        """Drop rows whose feature vector is duplicated. Prevents train/test leak
        when 5-tuple + timestamp ID cols are removed in ``clean()``."""
        feats = [c for c in df.columns if c != self.cfg.label_col]
        before = len(df)
        df = df.drop_duplicates(subset=feats, ignore_index=True)
        return df, before - len(df)

    # ── Stage 2: per-class split (rare/abundant) ─────────────────────────
    def split(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        aug = self.cfg.augment
        rare_th = aug.get("rare_threshold", 500)
        rare_r = aug.get("rare_test_ratio", 0.5)
        def_r = aug.get("default_test_ratio", 0.2)
        test_cap = aug.get("test_cap", 2000)
        seed = self.cfg.split_seed
        label_col = self.cfg.label_col

        train_parts, test_parts = [], []
        s = seed
        for lbl, grp in df.groupby(label_col):
            grp = grp.sample(frac=1.0, random_state=s).reset_index(drop=True)
            n = len(grp)
            ratio = rare_r if n <= rare_th else def_r
            n_test = int(round(n * ratio))
            tr, te = grp.iloc[n_test:], grp.iloc[:n_test]
            if test_cap and len(te) > test_cap:
                te = te.iloc[:test_cap]
            train_parts.append(tr)
            test_parts.append(te)
            kind = "rare " if n <= rare_th else "abund"
            logger.info("  class %s (%s, total=%7d) → train=%7d test=%7d (ratio=%g)",
                        lbl, kind, n, len(tr), len(te), ratio)
            s += 1
        train_df = pd.concat(train_parts).sample(
            frac=1.0, random_state=seed,
        ).reset_index(drop=True)
        test_df = pd.concat(test_parts).sample(
            frac=1.0, random_state=seed + 1,
        ).reset_index(drop=True)
        return train_df, test_df

    # ── Stage 4: cap abundant classes for SHAP/CovaS ──────────────────────
    def cap(self, df: pd.DataFrame, *, method: str = "random") -> pd.DataFrame:
        """Cap each abundant class to ``cfg.augment['shap_cap']`` rows.

        ``method`` ∈ {"random", "minibatch-kmeans"}. Per-class sampling uses
        a single shared seed (no shuffle, no bumping) to stay bit-exact with
        the original ``cap_for_shap.py`` recipe.
        """
        cap = self.cfg.augment.get("shap_cap", 5000)
        seed = self.cfg.split_seed
        label_col = self.cfg.label_col

        parts = []
        for lbl, grp in df.groupby(label_col):
            if len(grp) <= cap:
                parts.append(grp)
                logger.info("  class %s: %7d ≤ %d → keep full", lbl, len(grp), cap)
                continue
            if method == "random":
                sub = grp.sample(n=cap, random_state=seed).reset_index(drop=True)
                logger.info("  class %s: %7d → cap %d (random)", lbl, len(grp), cap)
            elif method == "minibatch-kmeans":
                from sklearn.cluster import MiniBatchKMeans
                from sklearn.metrics import pairwise_distances_argmin_min
                feat_cols = [c for c in grp.columns if c != label_col]
                X = grp[feat_cols].values.astype(np.float32)
                km = MiniBatchKMeans(n_clusters=cap, batch_size=10000,
                                     max_iter=100, n_init="auto",
                                     random_state=seed + int(lbl)).fit(X)
                idx, _ = pairwise_distances_argmin_min(km.cluster_centers_, X)
                idx = np.unique(idx.astype(np.int64))
                sub = grp.iloc[np.sort(idx)].reset_index(drop=True)
                logger.info("  class %s: %7d → cap %d (kmeans, %d uniq)",
                            lbl, len(grp), cap, len(idx))
            else:
                raise ValueError(f"Unknown cap method: {method}")
            parts.append(sub)

        return pd.concat(parts, ignore_index=True).sample(
            frac=1.0, random_state=seed,
        ).reset_index(drop=True)

    # ── Stage 5a: CovaS dead features ─────────────────────────────────────
    def covas_dead(self, df: pd.DataFrame) -> tuple[list[str], pd.DataFrame]:
        """Run CovaS on the configured class pair, return ``(dead_list, stats_df)``."""
        cov = self.cfg.covas
        pair_names = cov.get("pair")
        if not pair_names:
            raise ValueError("cfg.covas missing 'pair'")
        bins = cov.get("bins", 100)
        robust = bool(cov.get("robust", True))
        th = CovasThresholds(**(cov.get("thresholds") or {}))

        if not self.cfg.label_names:
            raise ValueError("cfg.label_names required to resolve CovaS pair")
        name_to_id = {n: i for i, n in enumerate(self.cfg.label_names)}
        try:
            id_a = name_to_id[pair_names[0]]
            id_b = name_to_id[pair_names[1]]
        except KeyError as exc:
            raise KeyError(f"CovaS pair {pair_names} not in label_names "
                           f"{list(self.cfg.label_names)}") from exc

        features = [c for c in df.columns if c != self.cfg.label_col]
        logger.info("CovaS pair %s(%d) vs %s(%d) on %d features",
                    pair_names[0], id_a, pair_names[1], id_b, len(features))
        stats = pair_feature_stats(df, self.cfg.label_col,
                                   id_a, id_b, features,
                                   bins=bins, robust=robust)
        dead = dead_features(stats, th)
        logger.info("Dead features (%d): %s", len(dead), dead)
        return dead, stats

    # ── Stage 5b: SHAP top-K∪ feature selection ───────────────────────────
    def shap_select(
        self, df: pd.DataFrame, *, drop: tuple[str, ...] = (),
    ) -> tuple[list[str], pd.DataFrame]:
        """Train RF + XGB, compute mean(|shap|), union of top-K from each model.

        ``drop`` is the CovaS dead-features list (excluded from candidate set).
        Returns ``(selected_features, ranking_df)``.
        """
        import shap
        from sklearn.model_selection import train_test_split
        from sklearn.utils.class_weight import compute_sample_weight

        from .models import RandomForestModel, XGBoostModel

        sh = self.cfg.shap
        top_k = sh.get("top_k", 20)
        val_size = sh.get("val_size", 0.1)
        seed = sh.get("random_state", self.cfg.split_seed)
        label_col = self.cfg.label_col

        features = [c for c in df.columns if c != label_col and c not in drop]
        logger.info("SHAP candidate features (after drop %d): %d",
                    len(drop), len(features))
        X = df[features].values.astype(np.float32)
        y = df[label_col].values.astype(int)

        Xtr, Xval, ytr, yval = train_test_split(
            X, y, test_size=val_size, random_state=seed, stratify=y,
        )

        logger.info("Training RF for SHAP …")
        rf = RandomForestModel().train(Xtr, ytr, cfg=sh.get("rf_cfg") or None)
        logger.info("Training XGB for SHAP …")
        sw_tr = compute_sample_weight("balanced", ytr)
        sw_val = compute_sample_weight("balanced", yval)
        xgb = XGBoostModel().train(Xtr, ytr, X_val=Xval, y_val=yval,
                                   cfg=sh.get("xgb_cfg") or None,
                                   sample_weight=sw_tr,
                                   sample_weight_eval=sw_val)

        rf_imp = mean_abs_shap(shap.TreeExplainer(rf._model).shap_values(Xtr))
        xgb_imp = mean_abs_shap(shap.TreeExplainer(xgb._model).shap_values(
            Xtr, check_additivity=False,
        ))

        rf_norm = rf_imp / (rf_imp.sum() + 1e-12)
        xgb_norm = xgb_imp / (xgb_imp.sum() + 1e-12)
        feats_arr = np.array(features)
        top_rf = feats_arr[np.argsort(-rf_norm)[:top_k]].tolist()
        top_xgb = feats_arr[np.argsort(-xgb_norm)[:top_k]].tolist()
        selected = sorted(set(top_rf) | set(top_xgb))

        ranking = pd.DataFrame({
            "feature": features,
            "shap_rf": rf_norm,
            "shap_xgb": xgb_norm,
            "combined": (rf_norm + xgb_norm) / 2,
            "in_top_rf": [f in top_rf for f in features],
            "in_top_xgb": [f in top_xgb for f in features],
            "selected": [f in selected for f in features],
        }).sort_values("combined", ascending=False)

        logger.info("SHAP: top-%d RF=%d, top-%d XGB=%d, UNION=%d",
                    top_k, len(top_rf), top_k, len(top_xgb), len(selected))
        return selected, ranking

    # ── Stage 6: project a df onto a feature subset ──────────────────────
    def apply_features(
        self, df: pd.DataFrame, features,
    ) -> pd.DataFrame:
        feats = [f for f in features if f in df.columns]
        return df[feats + [self.cfg.label_col]]


def mean_abs_shap(shap_values) -> np.ndarray:
    """Reduce shap_values to mean(|shap|) per feature across samples + classes."""
    if isinstance(shap_values, list):
        per_class = [np.abs(s).mean(axis=0) for s in shap_values]
        return np.mean(per_class, axis=0)
    arr = np.asarray(shap_values)
    if arr.ndim == 3:
        return np.abs(arr).mean(axis=0).mean(axis=-1)
    return np.abs(arr).mean(axis=0)
