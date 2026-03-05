"""Prepare augmented training data for adversarial training via TVAE.

Reads the original (pre-augmentation) training CSV, fits a TVAESynthesizer
per class for under-represented classes, then saves the combined dataset
(original + synthetic) to adv_samples/adv_training/train_at.csv.

This augmented dataset is then consumed by adv_train_dl.py as training data
for online adversarial training.

Pipeline position:
    DS/train_shap_66.csv  (original)
        ↓  prepare_adv_data.py
    adv_samples/adv_training/train_at.csv  (TVAE-augmented)
        ↓  adv_train_dl.py
    models/framework_lstm_TVAE_at_pgd.pth  (robust model)

Usage:
    # Default: augment all classes to 10 000 samples each
    python prepare_adv_data.py

    # Custom sample cap and specific labels to augment
    python prepare_adv_data.py --sample-max 5000 --labels 7 8 9 10

    # Custom input / output paths
    python prepare_adv_data.py --train-csv /data/train.csv --out-csv /out/train_at.csv

    # Disable CUDA (CPU-only TVAE)
    python prepare_adv_data.py --no-cuda
"""

import os
import sys
import argparse
import warnings

warnings.filterwarnings('ignore')

# ── Path bootstrap ──────────────────────────────────────────────────────────────
_HERE  = os.path.dirname(os.path.realpath(__file__))
_FOAMI = os.path.dirname(os.path.dirname(_HERE))   # foami+/
sys.path.insert(0, _FOAMI)

from utils.paths import setup_paths, AT_TRAIN_CSV, TRAIN_ORIG_CSV
setup_paths()

from utils.logging import setup_logging, get_logger

import pandas as pd

logger = get_logger(__name__)


# ── TVAE augmentation ──────────────────────────────────────────────────────────

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


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="TVAE data augmentation for adversarial training"
    )
    parser.add_argument('--train-csv', default=TRAIN_ORIG_CSV,
                        help=f"Input training CSV (default: {TRAIN_ORIG_CSV})")
    parser.add_argument('--out-csv', default=AT_TRAIN_CSV,
                        help=f"Output augmented CSV (default: {AT_TRAIN_CSV})")
    parser.add_argument('--sample-max', type=int, default=10_000,
                        help="Target number of samples per class (default: 10000)")
    parser.add_argument('--labels', type=int, nargs='*', default=None,
                        help="Label values to augment (default: all classes)")
    parser.add_argument('--no-cuda', action='store_true',
                        help="Disable CUDA — train TVAE on CPU")
    parser.add_argument('--log-level', default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'])
    args = parser.parse_args()

    setup_logging(args.log_level)

    if not os.path.exists(args.train_csv):
        raise SystemExit(f"Train CSV not found: {args.train_csv}")

    label_col = 'Label'
    use_cuda  = not args.no_cuda

    logger.info(f"[+] Input  : {args.train_csv}")
    logger.info(f"[+] Output : {args.out_csv}")
    logger.info(f"[+] sample_max={args.sample_max}, cuda={use_cuda}")

    df_train = pd.read_csv(args.train_csv, low_memory=False)
    logger.info(f"[+] Loaded: {df_train.shape}")

    all_labels = sorted(df_train[label_col].unique().tolist())
    labels     = args.labels if args.labels is not None else all_labels
    logger.info(f"[+] Classes to augment: {labels}")

    # ── Class distribution before augmentation ─────────────────────────────
    dist = df_train[label_col].value_counts().sort_index()
    logger.info(f"[+] Class distribution (before):\n{dist.to_string()}")

    # ── TVAE augmentation ──────────────────────────────────────────────────
    df_aug = tvae_augment(
        df_train  = df_train,
        label_col = label_col,
        sample_max= args.sample_max,
        labels    = labels,
        use_cuda  = use_cuda,
    )

    # ── Class distribution after augmentation ──────────────────────────────
    dist_after = df_aug[label_col].value_counts().sort_index()
    logger.info(f"[+] Class distribution (after):\n{dist_after.to_string()}")
    logger.info(f"[+] Final shape: {df_aug.shape}")

    # ── Save ───────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    df_aug.to_csv(args.out_csv, index=False)
    logger.info(f"[+] Saved → {args.out_csv}")


if __name__ == '__main__':
    main()
