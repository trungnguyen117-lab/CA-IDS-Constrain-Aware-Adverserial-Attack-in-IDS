"""Generate adversarial samples for DL (raw space via InputNorm) and ML (BB direct)."""

import argparse
import os
import sys

# Path bootstrap — _CLI must be first so cli/utils/ shadows script/utils/
_HERE = os.path.dirname(os.path.abspath(__file__))
_CLI = os.path.dirname(_HERE)
_IEC = os.path.dirname(_CLI)
sys.path.insert(0, _IEC)
sys.path.insert(0, os.path.join(_IEC, "script"))
sys.path.insert(0, _CLI)

from utils.logging import setup_logging, get_logger
from utils.constants import (
    DL_TARGETS, ALL_TARGETS, ALL_ATTACKS,
    ATTACK_COMPAT, ATTACK_GENERATORS,
)
from utils.paths import (
    model_path, adv_eval_dir, adv_train_dir,
    load_attack_config, load_adv_training_config,
    set_version, model_stem,
)
from utils.loaders import load_dataset
from utils.masking import get_mutate_indices
from model import get_model

logger = get_logger(__name__)


# ── Helpers ──────────────────────────────────────────────────────────────────


def _get_input_metadata(feature_names):
    return {"feature_names": feature_names, "label_column": "Label"}


def _save_adv(df_adv, out_dir, filename):
    """Save adversarial DataFrame."""
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, filename)
    df_adv.to_csv(out_path, index=False)
    logger.info(f"Saved {len(df_adv)} adv samples -> {out_path}")
    return out_path


def _get_generator(attack_name, classifier, config):
    """Instantiate attack generator from art_generator."""
    import art_generator as ag
    cls_name = ATTACK_GENERATORS[attack_name]
    cls = getattr(ag, cls_name)
    return cls(classifier, generator_params=config)


# ── DL Attack (raw space — InputNorm embedded in model) ───────────────────


def generate_dl(target, attack_name, source, device, X, y, feature_names, mutate_indices):
    """Generate adversarial for DL model in raw space.

    InputNorm is embedded in the model architecture, so:
      1. Load model → wrap for ART (raw input, InputNorm handles normalization)
      2. art_generator.generate(X_raw, y, ...) → X_adv_raw
      3. Save raw CSV directly (no inverse_transform needed)
    """
    # Load model via registry
    m = get_model(target)
    ext = ".pth"
    m_inst = type(m).load(model_path(f"{model_stem(target)}{ext}"), device=device)

    # Wrap for ART — model accepts raw input (InputNorm embedded)
    art_clf = m_inst.wrap_for_art(X, device=device)

    # Load attack config
    if source == "train":
        atk_cfg = load_adv_training_config(attack_name)
    else:
        atk_cfg = load_attack_config(attack_name)
    logger.info(f"Attack config ({attack_name}): {atk_cfg}")

    # Generate in raw space
    gen = _get_generator(attack_name, art_clf, atk_cfg)
    metadata = _get_input_metadata(feature_names)
    df_adv = gen.generate(X, y, metadata, mutate_indices)

    # Save: adv_{eval|training}/{target}_sc/{target}_{attack}_adv.csv
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
    """Generate adversarial for ML model via direct black-box attack.

    Follows notebook pattern:
      1. Load model → CatBoostARTClassifier / SklearnClassifier
      2. art_generator.generate(X_raw, y, ...) → X_adv_raw
      3. Save raw CSV
    """
    # Load model via registry
    ext = ".pkl"
    m_inst = type(get_model(target)).load(model_path(f"{model_stem(target)}{ext}"))
    art_clf = m_inst.wrap_for_art(X)

    # Load attack config
    if source == "train":
        atk_cfg = load_adv_training_config(attack_name)
    else:
        atk_cfg = load_attack_config(attack_name)
    logger.info(f"Attack config ({attack_name}): {atk_cfg}")

    # Generate in raw space
    gen = _get_generator(attack_name, art_clf, atk_cfg)
    metadata = _get_input_metadata(feature_names)
    df_adv = gen.generate(X, y, metadata, mutate_indices)

    # Save: adv_{eval|training}/{target}/{target}_{attack}_adv.csv
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
                        choices=["cpu", "cuda", "auto", "mps"])
    parser.add_argument("--max-samples", "-n", type=int, default=None,
                        help="Stratified subsample size (useful for slow BB attacks)")
    parser.add_argument("--version", "-V", default="v1",
                        help="Version tag for adv samples/models (default: v1)")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    setup_logging(args.log_level)
    set_version(args.version)

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

    # Load data once via shared loader
    key = "train_tvae" if args.source == "train" else "test"
    df, X, y, feature_names = load_dataset(key, max_samples=args.max_samples)

    # Always compute mutate_indices from TRAIN to ensure consistency
    # (test may have different unique-value counts due to smaller size)
    df_train_ref, *_ = load_dataset("train")
    mutate_indices = get_mutate_indices(df_train_ref)
    logger.info(f"Data: {X.shape}, Protected features: {len(mutate_indices)}")

    for target in targets:
        compatible = ATTACK_COMPAT[target]
        for attack_name in attacks:
            if attack_name not in compatible:
                logger.warning(f"Skipping {attack_name} for {target} (incompatible)")
                continue

            logger.info(f"{'='*60}")
            logger.info(f"Target: {target.upper()} | Attack: {attack_name.upper()} | Source: {args.source}")
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
