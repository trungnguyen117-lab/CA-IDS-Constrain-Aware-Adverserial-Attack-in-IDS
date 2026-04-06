"""Evaluate DAE Embedding defense for IEC-104 ensemble.

Approach: Use DAE encoder as preprocessing — map input from 66-feature space
to 64-dim bottleneck embedding. Train classifiers on embeddings.

Hypothesis: adversarial perturbations crafted in 66-feature space won't
survive compression through the bottleneck. The encoder learned to map
clean data patterns; adversarial deviations get "squeezed out".

4-way comparison on ENSEMBLE level:
  Ensemble (original) | Ensemble+AT | DAE-Embed Ensemble | DAE-Embed+AT Ensemble

Usage:
    python defense/dae_embedding/evaluate_dae_embedding.py --device mps --at both
    python defense/dae_embedding/evaluate_dae_embedding.py --device mps --at both \
        --bottleneck-dim 32 48 64 96
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
import yaml
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score

# Path bootstrap
_HERE = os.path.dirname(os.path.abspath(__file__))
_IEC = os.path.dirname(os.path.dirname(_HERE))
_CLI = os.path.join(_IEC, "cli")
_DAE = os.path.join(_IEC, "defense", "dae")
sys.path.insert(0, _IEC)
sys.path.insert(0, os.path.join(_IEC, "script"))
sys.path.insert(0, _CLI)
sys.path.insert(0, _DAE)

from utils.logging import setup_logging, get_logger
from utils.constants import (
    ALL_TARGETS, DL_TARGETS, TREE_TARGETS,
    BB_ATTACKS, WB_ATTACKS,
)
from utils.evaluation import report_metrics, compute_asr
from utils.loaders import load_dataset
from utils.paths import model_path, adv_eval_dir
from utils.ensemble import weighted_soft_voting, DEFAULT_WEIGHTS
from model import MODEL_REGISTRY

from dae_model import load_checkpoint as load_dae_checkpoint

logger = get_logger(__name__)

_ADV_TRAIN_ROOT = os.path.join(_IEC, "adv_samples", "adv_training")


# ── DAE Encoder ──────────────────────────────────────────────────────────────


class DAEEncoder:
    """Wraps trained DAE — extracts encoder to produce bottleneck embeddings."""

    def __init__(self, dae_path, device="cpu"):
        self.device = device
        model, scaler, cfg = load_dae_checkpoint(dae_path, device=device)
        self.model = model
        self.scaler = scaler
        self.cfg = cfg
        self.bottleneck_dim = cfg["bottleneck_dim"]
        logger.info(
            f"DAE encoder loaded: {cfg['arch']}, "
            f"bottleneck_dim={self.bottleneck_dim}"
        )

    @torch.no_grad()
    def encode(self, X):
        """Encode features → bottleneck embedding.

        Args:
            X: ndarray (n_samples, n_features) in original scale

        Returns:
            Z: ndarray (n_samples, bottleneck_dim)
        """
        X_scaled = self.scaler.transform(X).astype(np.float32)
        x_t = torch.from_numpy(X_scaled).to(self.device)

        self.model.eval()
        m = self.model

        # Extract encoder path based on architecture
        if hasattr(m, "enc1"):
            # ResidualDAE
            e1 = m.enc1(x_t)
            for blk in m.enc1_res:
                e1 = blk(e1)
            e2 = m.enc2(e1)
            for blk in m.enc2_res:
                e2 = blk(e2)
            z = m.enc_bottleneck(e2)
        elif hasattr(m, "encoder"):
            # VanillaDAE
            z = m.encoder(x_t)
        else:
            raise RuntimeError(f"Unknown DAE architecture: {type(m)}")

        return z.cpu().numpy()


# ── Embedding Classifier Ensemble ────────────────────────────────────────────


class EmbeddingEnsemble:
    """Train and evaluate an ensemble of classifiers on DAE embeddings."""

    def __init__(self):
        self.models = {}
        self.weights = {
            "catboost": 0.30,
            "rf": 0.35,
            "xgb": 0.35,
        }

    def fit(self, Z_train, y_train):
        """Train classifiers on embedding space."""
        logger.info(f"Training embedding ensemble on {Z_train.shape}")

        self.n_classes = int(y_train.max()) + 1

        # CatBoost
        cb = CatBoostClassifier(
            iterations=1000, depth=6, learning_rate=0.1,
            loss_function="MultiClass", verbose=0,
            random_seed=42, task_type="CPU",
        )
        cb.fit(Z_train, y_train)
        self.models["catboost"] = cb
        logger.info("  CatBoost trained")

        # Random Forest
        rf = RandomForestClassifier(
            n_estimators=500, max_depth=None, min_samples_leaf=2,
            random_state=42, n_jobs=-1,
        )
        rf.fit(Z_train, y_train)
        self.models["rf"] = rf
        logger.info("  RF trained")

        # XGBoost
        xgb = XGBClassifier(
            n_estimators=500, max_depth=6, learning_rate=0.1,
            random_state=42, n_jobs=-1,
            objective="multi:softprob", num_class=self.n_classes,
            eval_metric="mlogloss", verbosity=0,
        )
        xgb.fit(Z_train, y_train)
        self.models["xgb"] = xgb
        logger.info("  XGBoost trained")

    def predict_proba(self, Z):
        """Get ensemble probability predictions."""
        probas = {}
        for name, m in self.models.items():
            p = m.predict_proba(Z)
            # Ensure all models output same number of classes
            if p.shape[1] < self.n_classes:
                padded = np.zeros((len(Z), self.n_classes), dtype=np.float32)
                padded[:, :p.shape[1]] = p
                p = padded
            probas[name] = p.astype(np.float32)
        return probas

    def predict(self, Z):
        """Get ensemble predictions via weighted soft voting."""
        probas = self.predict_proba(Z)
        _, y_pred = weighted_soft_voting(probas, weights=self.weights)
        return y_pred


# ── Adversarial CSV resolution (reuse from kmeans eval) ─────────────────────


def _adv_csv_path(target, attack):
    if target in DL_TARGETS:
        return adv_eval_dir(f"{target}_sc", f"{target}_{attack}_adv.csv")
    else:
        if attack in BB_ATTACKS:
            return adv_eval_dir(target, f"{target}_{attack}_adv.csv")
        else:
            return adv_eval_dir("resdnn_sc", f"resdnn_{attack}_adv.csv")


def _discover_attacks():
    candidate_attacks = set()
    for dl_target in DL_TARGETS:
        d = adv_eval_dir(f"{dl_target}_sc")
        if not os.path.isdir(d):
            continue
        for fname in os.listdir(d):
            if not fname.endswith("_adv.csv") or "surrogate" in fname:
                continue
            basename = fname.replace("_adv.csv", "")
            prefix = dl_target + "_"
            atk = basename[len(prefix):] if basename.startswith(prefix) else basename
            candidate_attacks.add(atk)

    for tree_target in TREE_TARGETS:
        d = adv_eval_dir(tree_target)
        if not os.path.isdir(d):
            continue
        for fname in os.listdir(d):
            if not fname.endswith("_adv.csv") or "surrogate" in fname:
                continue
            basename = fname.replace("_adv.csv", "")
            prefix = tree_target + "_"
            atk = basename[len(prefix):] if basename.startswith(prefix) else basename
            candidate_attacks.add(atk)

    attacks = []
    for atk in sorted(candidate_attacks):
        all_exist = True
        for target in ALL_TARGETS:
            if not os.path.isfile(_adv_csv_path(target, atk)):
                all_exist = False
                break
        if all_exist:
            attacks.append(atk)
    return attacks


# ── Load original FOAMI models ──────────────────────────────────────────────


def _load_model(target, at=False, device="cpu"):
    suffix = "_at" if at else ""
    ext = ".pkl" if target in TREE_TARGETS else ".pth"
    path = model_path(f"framework_{target}_TVAE{suffix}{ext}", at=at)
    return MODEL_REGISTRY[target].load(path, device=device)


def _load_all_models(at=False, device="cpu"):
    models = {}
    for target in ALL_TARGETS:
        try:
            models[target] = _load_model(target, at=at, device=device)
            logger.info(f"Loaded {target} {'(AT)' if at else '(baseline)'}")
        except Exception as e:
            logger.error(f"Failed loading {target}: {e}")
    return models


# ── Evaluate original ensemble (for comparison) ─────────────────────────────


def evaluate_original_ensemble(models, X_test, y_test, feature_names, attacks,
                                at=False, device="cpu"):
    """Standard FOAMI ensemble evaluation."""
    variant = "Ensemble+AT" if at else "Ensemble"

    clean_probas = {}
    for name, m in models.items():
        art_clf = m.wrap_for_art(X_test, device=device)
        proba = m.art_predict_proba(art_clf, X_test)
        clean_probas[name] = proba

    _, y_ens_clean = weighted_soft_voting(clean_probas)
    clean_metrics = report_metrics(f"{variant} clean", y_test, y_ens_clean)
    results = {"clean": clean_metrics}

    for atk in attacks:
        try:
            adv_probas = {}
            for target in models:
                csv_path = _adv_csv_path(target, atk)
                df_adv = pd.read_csv(csv_path)
                X_adv = df_adv[feature_names].values.astype(np.float32)
                art_clf = models[target].wrap_for_art(X_adv, device=device)
                proba = models[target].art_predict_proba(art_clf, X_adv)
                adv_probas[target] = proba

            _, y_ens_adv = weighted_soft_voting(adv_probas)
            metrics = report_metrics(f"{variant} {atk}", y_test, y_ens_adv)
            asr = compute_asr(y_test, y_ens_clean, y_ens_adv)
            metrics["asr"] = asr
            results[atk] = metrics
        except Exception as e:
            logger.error(f"Failed {variant} on {atk}: {e}")

    return results


# ── Evaluate DAE Embedding ensemble ──────────────────────────────────────────


def evaluate_embedding_ensemble(encoder, emb_ensemble, X_test, y_test,
                                 feature_names, attacks, variant="DAE-Embed"):
    """Evaluate DAE embedding ensemble on clean + adversarial data."""
    logger.info(f"{'='*60}")
    logger.info(f"Evaluating: {variant}")
    logger.info(f"{'='*60}")

    # Clean
    Z_test = encoder.encode(X_test)
    y_clean = emb_ensemble.predict(Z_test)
    clean_metrics = report_metrics(f"{variant} clean", y_test, y_clean)
    results = {"clean": clean_metrics}

    if not attacks:
        return results

    for atk in attacks:
        try:
            # For embedding ensemble, we use the SAME adversarial CSV
            # (pick one representative — resdnn for WB, or target-specific for BB)
            # Actually: use each target's adversarial CSV and take one
            # Since all models in embedding ensemble see the same input,
            # we just need ONE adversarial set per attack.
            # Use first available target's adv CSV.
            csv_path = _adv_csv_path(ALL_TARGETS[0], atk)
            df_adv = pd.read_csv(csv_path)
            X_adv = df_adv[feature_names].values.astype(np.float32)

            Z_adv = encoder.encode(X_adv)
            y_adv = emb_ensemble.predict(Z_adv)

            metrics = report_metrics(f"{variant} {atk}", y_test, y_adv)
            asr = compute_asr(y_test, y_clean, y_adv)
            metrics["asr"] = asr
            logger.info(f"{'':>25s}  ASR={asr:6.2f}%")
            results[atk] = metrics
        except Exception as e:
            logger.error(f"Failed {variant} on {atk}: {e}")

    return results


# ── Adversarial Training on Embeddings ────────────────────────────────────────


def _collect_adv_training_data(feature_names, max_samples=30000):
    """Collect adversarial training CSVs, subsample to max_samples.

    Loads all CSVs then stratified-subsamples to keep training fast
    while maintaining attack diversity.

    Returns (X_adv_all, y_adv_all) as numpy arrays.
    """
    all_X = []
    all_y = []

    for root, dirs, files in os.walk(_ADV_TRAIN_ROOT):
        for fname in sorted(files):
            if not fname.endswith("_train_adv.csv"):
                continue
            path = os.path.join(root, fname)
            try:
                df = pd.read_csv(path)
                X = df[feature_names].values.astype(np.float32)
                y = df["Label"].values.astype(int)
                all_X.append(X)
                all_y.append(y)
                logger.debug(f"  Loaded {fname}: {len(X)} samples")
            except Exception as e:
                logger.warning(f"  Skipping {fname}: {e}")

    if not all_X:
        return None, None

    X_all = np.concatenate(all_X, axis=0)
    y_all = np.concatenate(all_y, axis=0)
    logger.info(f"Adversarial training data: {len(all_X)} files, "
                f"{X_all.shape[0]} total samples")

    # Subsample if too large
    if len(X_all) > max_samples:
        rng = np.random.RandomState(42)
        idx = rng.choice(len(X_all), size=max_samples, replace=False)
        X_all = X_all[idx]
        y_all = y_all[idx]
        logger.info(f"Subsampled to {max_samples} adversarial samples")

    return X_all, y_all


def train_at_embedding_ensemble(encoder, X_train, y_train, feature_names):
    """Train embedding ensemble with adversarial training.

    1. Encode clean training data → Z_clean
    2. Collect adversarial training CSVs → encode → Z_adv
    3. Train classifiers on Z_clean + Z_adv
    """
    logger.info("Training DAE-Embed+AT ensemble...")

    # Encode clean training data
    Z_clean = encoder.encode(X_train)

    # Collect and encode adversarial training data
    X_adv, y_adv = _collect_adv_training_data(feature_names)
    if X_adv is None:
        logger.error("No adversarial training data found!")
        return None

    Z_adv = encoder.encode(X_adv)
    logger.info(f"Adversarial embeddings: {X_adv.shape} → {Z_adv.shape}")

    # Combine clean + adversarial embeddings
    Z_combined = np.concatenate([Z_clean, Z_adv], axis=0)
    y_combined = np.concatenate([y_train, y_adv], axis=0)

    # Shuffle
    rng = np.random.RandomState(42)
    idx = rng.permutation(len(Z_combined))
    Z_combined = Z_combined[idx]
    y_combined = y_combined[idx]

    logger.info(f"Combined training: {Z_combined.shape[0]} samples "
                f"(clean={len(Z_clean)}, adv={len(Z_adv)})")

    # Train ensemble on combined data
    emb_at = EmbeddingEnsemble()
    emb_at.fit(Z_combined, y_combined)
    return emb_at


# ── Rich Table ───────────────────────────────────────────────────────────────


def _print_comparison(results_dict, attacks):
    from rich.table import Table
    from rich.console import Console

    labels = list(results_dict.keys())
    t = Table(
        title="DAE Embedding Ensemble Defense",
        show_lines=True, title_style="bold cyan",
    )
    t.add_column("Metric", style="bold", min_width=18)
    for label in labels:
        t.add_column(label, justify="right", min_width=12)

    def _fmt(val):
        return f"{val:.2f}%" if val is not None else "-"

    for key, metric in [("acc", "Acc (clean)"), ("f1", "F1 (clean)")]:
        row = [metric]
        for label in labels:
            row.append(_fmt(results_dict[label].get("clean", {}).get(key)))
        t.add_row(*row)

    t.add_section()

    for atk in attacks:
        row = [f"ASR ({atk})"]
        for label in labels:
            row.append(_fmt(results_dict[label].get(atk, {}).get("asr")))
        t.add_row(*row)

    t.add_section()

    for atk in attacks:
        row = [f"Acc ({atk})"]
        for label in labels:
            row.append(_fmt(results_dict[label].get(atk, {}).get("acc")))
        t.add_row(*row)

    Console().print(t)
    Console().print()


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate DAE Embedding Ensemble defense")
    parser.add_argument("--at", default="both",
                        choices=["false", "true", "both"])
    parser.add_argument("--device", "-d", default="auto",
                        choices=["cpu", "cuda", "mps", "auto"])
    parser.add_argument("--dae-path", default=None,
                        help="Path to trained DAE checkpoint")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    setup_logging(args.log_level)

    if args.device == "auto":
        if torch.cuda.is_available():
            args.device = "cuda"
        elif torch.backends.mps.is_available():
            args.device = "mps"
        else:
            args.device = "cpu"
    logger.info(f"Device: {args.device}")

    # Load DAE encoder
    dae_path = args.dae_path or os.path.join(
        _DAE, "models", "dae_iec104_residual.pth")
    encoder = DAEEncoder(dae_path, device=args.device)

    # Load data
    _, X_train, y_train, _ = load_dataset("train_tvae")
    _, X_test, y_test, feature_names = load_dataset("test")

    # Encode training data
    logger.info("Encoding training data through DAE encoder...")
    Z_train = encoder.encode(X_train)
    logger.info(f"Embeddings: {X_train.shape} → {Z_train.shape}")

    # Train embedding ensemble
    emb_ensemble = EmbeddingEnsemble()
    emb_ensemble.fit(Z_train, y_train)

    # Discover attacks
    attacks = _discover_attacks()
    if not attacks:
        logger.error("No adversarial attacks found.")
        sys.exit(1)
    logger.info(f"Found {len(attacks)} attacks: {attacks}")

    all_results = {}

    # --- Original ensemble (baseline) ---
    models_base = _load_all_models(at=False, device=args.device)
    if len(models_base) >= len(ALL_TARGETS):
        all_results["Ensemble"] = evaluate_original_ensemble(
            models_base, X_test, y_test, feature_names, attacks,
            device=args.device,
        )

    # --- Original ensemble + AT ---
    if args.at in ("true", "both"):
        models_at = _load_all_models(at=True, device=args.device)
        if len(models_at) >= len(ALL_TARGETS):
            all_results["Ensemble+AT"] = evaluate_original_ensemble(
                models_at, X_test, y_test, feature_names, attacks,
                at=True, device=args.device,
            )

    # --- DAE Embedding ensemble ---
    all_results["DAE-Embed"] = evaluate_embedding_ensemble(
        encoder, emb_ensemble, X_test, y_test, feature_names, attacks,
        variant="DAE-Embed",
    )

    # --- DAE Embedding ensemble + AT (adversarial training in embedding space) ---
    if args.at in ("true", "both"):
        emb_at = train_at_embedding_ensemble(
            encoder, X_train, y_train, feature_names)
        if emb_at is not None:
            all_results["DAE-Embed+AT"] = evaluate_embedding_ensemble(
                encoder, emb_at, X_test, y_test, feature_names, attacks,
                variant="DAE-Embed+AT",
            )

    _print_comparison(all_results, attacks)


if __name__ == "__main__":
    main()
