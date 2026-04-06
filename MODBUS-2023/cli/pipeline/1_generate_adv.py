"""Generate adversarial samples for DL (WB scaled-space) and ML (BB direct)."""

import argparse
import os
import sys
import numpy as np
import pandas as pd

# Path bootstrap
_HERE = os.path.dirname(os.path.abspath(__file__))
_CLI = os.path.dirname(_HERE)
_MODBUS = os.path.dirname(_CLI)
sys.path.insert(0, _MODBUS)
sys.path.insert(0, _CLI)

from utils.logging import setup_logging, get_logger
from utils.constants import (
    DL_TARGETS, ALL_TARGETS, ALL_ATTACKS,
    ATTACK_COMPAT, ATTACK_GENERATORS,
)
from utils.paths import (
    model_path, adv_eval_dir, adv_train_dir,
    load_attack_config, load_adv_training_config,
)
from utils.loaders import load_dataset
from utils.masking import get_mutate_indices
from model import get_model

logger = get_logger(__name__)


def _get_input_metadata(feature_names):
    return {"feature_names": feature_names, "label_column": "Label"}


# Keys in YAML that map to BatchedAttackGenerator kwargs
_OUTER_KEYS = {
    "outer_batch_size": "batch_size",
    "max_retries": "max_retries",
    "timeout": "timeout",
    "verbose_every": "verbose",
}


def _batch_kwargs(attack_name: str, atk_cfg: dict) -> dict:
    """Return batching kwargs from YAML config only."""
    if atk_cfg is None:
        return {}

    result = {}
    for yaml_key, kwarg_key in _OUTER_KEYS.items():
        if yaml_key in atk_cfg:
            result[kwarg_key] = atk_cfg[yaml_key]

    return result


def _save_adv(df_adv, out_dir, filename):
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, filename)
    df_adv.to_csv(out_path, index=False)
    logger.info(f"Saved {len(df_adv)} adv samples → {out_path}")
    return out_path


def _get_generator(attack_name, classifier, config):
    import art_generator as ag
    # Filter out outer batch keys before passing to ART
    art_config = {k: v for k, v in config.items() if k not in _OUTER_KEYS}
    cls_name = ATTACK_GENERATORS[attack_name]
    cls = getattr(ag, cls_name)
    return cls(classifier, generator_params=art_config)


# ── DL Attack (scaled-space) ────────────────────────────────────────────────


def generate_dl(target, attack_name, source, device, X, y, feature_names, mutate_indices):
    """Generate adversarial for DL model in scaled-space, inverse-transform to raw."""
    m = get_model(target)
    ext = ".pth"
    m_inst = type(m).load(model_path(f"framework_{target}_TabDiff{ext}"), device=device)

    scaler = m_inst.scaler
    X_sc = scaler.transform(X).astype(np.float64)
    clip_values = (float(X_sc.min()), float(X_sc.max()))

    art_clf = m_inst.wrap_for_art(X_sc, raw=False, device=device)

    if source == "train":
        atk_cfg = load_adv_training_config(attack_name)
    else:
        atk_cfg = load_attack_config(attack_name)
    logger.info(f"Attack config ({attack_name}): {atk_cfg}")

    gen = _get_generator(attack_name, art_clf, atk_cfg)
    metadata = _get_input_metadata(feature_names)
    batch_kwargs = _batch_kwargs(attack_name, atk_cfg)
    df_adv_sc = gen.generate(X_sc, y, metadata, mutate_indices, **batch_kwargs)

    X_adv_sc = df_adv_sc[feature_names].values.astype(np.float32)
    X_adv_raw = scaler.inverse_transform(X_adv_sc).astype(np.float32)
    df_adv = pd.DataFrame(X_adv_raw, columns=feature_names)
    df_adv["Label"] = y

    if source == "train":
        out_dir = adv_train_dir(f"{target}_sc")
        filename = f"{target}_{attack_name}_train_adv.csv"
    else:
        out_dir = adv_eval_dir(f"{target}_sc")
        filename = f"{target}_{attack_name}_adv.csv"

    _save_adv(df_adv, out_dir, filename)
    return df_adv


# ── ML Attack (direct BB) ───────────────────────────────────────────────────


def generate_ml(target, attack_name, source, X, y, feature_names, mutate_indices):
    """Generate adversarial for ML model via direct black-box attack."""
    ext = ".pkl"
    m_inst = type(get_model(target)).load(model_path(f"framework_{target}_TabDiff{ext}"))
    art_clf = m_inst.wrap_for_art(X)

    if source == "train":
        atk_cfg = load_adv_training_config(attack_name)
    else:
        atk_cfg = load_attack_config(attack_name)
    logger.info(f"Attack config ({attack_name}): {atk_cfg}")

    gen = _get_generator(attack_name, art_clf, atk_cfg)
    metadata = _get_input_metadata(feature_names)
    batch_kwargs = _batch_kwargs(attack_name, atk_cfg)
    df_adv = gen.generate(X, y, metadata, mutate_indices, **batch_kwargs)

    if source == "train":
        out_dir = adv_train_dir(target)
        filename = f"{target}_{attack_name}_train_adv.csv"
    else:
        out_dir = adv_eval_dir(target)
        filename = f"{target}_{attack_name}_adv.csv"

    _save_adv(df_adv, out_dir, filename)
    return df_adv


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Generate adversarial samples")
    parser.add_argument("--target", "-t", nargs="+", required=True,
                        choices=ALL_TARGETS + ["all"],
                        help="Target models to attack")
    parser.add_argument("--attack", "-a", nargs="+", required=True,
                        choices=ALL_ATTACKS + ["all"],
                        help="Attacks to run")
    parser.add_argument("--source", "-s", required=True,
                        choices=["train", "test"],
                        help="Data source: train (for AT) or test (for eval)")
    parser.add_argument("--device", "-d", default="cpu",
                        choices=["cpu", "cuda", "mps", "auto"])
    parser.add_argument("--max-samples", "-n", type=int, default=None,
                        help="Stratified subsample size")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    setup_logging(args.log_level)

    if args.device == "auto":
        import torch
        if torch.cuda.is_available():
            args.device = "cuda"
        elif torch.backends.mps.is_available():
            args.device = "mps"
        else:
            args.device = "cpu"

    targets = ALL_TARGETS if "all" in args.target else args.target
    attacks = ALL_ATTACKS if "all" in args.attack else args.attack

    key = "train_tvae" if args.source == "train" else "test"
    df, X, y, feature_names = load_dataset(key, max_samples=args.max_samples)
    mutate_indices = get_mutate_indices(df)
    logger.info(f"Data: {X.shape}, Protected features: {len(mutate_indices)}")

    for target in targets:
        compatible = ATTACK_COMPAT[target]
        for attack_name in attacks:
            if attack_name not in compatible:
                logger.warning(f"Skipping {attack_name} for {target} (incompatible)")
                continue

            logger.info(f"{'='*60}")
            logger.info(
                f"Target: {target.upper()} | Attack: {attack_name.upper()} "
                f"| Source: {args.source}")
            logger.info(f"{'='*60}")

            try:
                if target in DL_TARGETS:
                    generate_dl(target, attack_name, args.source, args.device,
                                X, y, feature_names, mutate_indices)
                else:
                    generate_ml(target, attack_name, args.source,
                                X, y, feature_names, mutate_indices)
            except Exception as e:
                logger.error(f"Failed {target}/{attack_name}: {e}", exc_info=True)


if __name__ == "__main__":
    main()
