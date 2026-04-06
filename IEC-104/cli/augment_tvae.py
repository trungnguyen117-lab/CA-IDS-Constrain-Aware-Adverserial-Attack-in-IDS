"""
TVAE augmentation — generate synthetic samples evenly across all classes.

Usage:
    python augment_tvae.py --total-samples 12000
    python augment_tvae.py --total-samples 12000 --epochs 300
    python augment_tvae.py --total-samples 12000 --no-eval
"""

import argparse
import os

import numpy as np
import pandas as pd
import sklearn.metrics
from sdv.metadata import Metadata
from sdv.single_table import TVAESynthesizer
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

import warnings
warnings.filterwarnings("ignore")

from utils.paths import get_path, cfg


def parse_args():
    parser = argparse.ArgumentParser(description='TVAE augmentation')
    parser.add_argument('--total-samples', type=int, required=True,
                        help='Total synthetic samples (divided evenly across all classes)')
    parser.add_argument('--epochs', type=int, default=256,
                        help='TVAE training epochs (default: 256)')
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--no-eval', action='store_true',
                        help='Skip XGB evaluation after augmentation')
    parser.add_argument('--cuda', action='store_true', default=True,
                        help='Use CUDA for TVAE training (default: True)')
    parser.add_argument('--no-cuda', action='store_true',
                        help='Disable CUDA')
    return parser.parse_args()


def main():
    args = parse_args()
    use_cuda = args.cuda and not args.no_cuda

    # -- Load data --
    train_df = pd.read_csv(get_path('train'), low_memory=False)
    print(f"Train: {train_df.shape}")

    all_labels = sorted(train_df['Label'].unique().tolist())
    n_classes = len(all_labels)
    samples_per_class = args.total_samples // n_classes

    print(f"Augmenting {n_classes} classes, {samples_per_class} samples/class "
          f"(total ~ {samples_per_class * n_classes})")

    # -- Train TVAE per class & generate --
    synthetic_dfs = []
    for label in all_labels:
        df_label = train_df[train_df["Label"] == label]
        current_count = len(df_label)
        n_to_generate = max(0, samples_per_class - current_count)

        if n_to_generate == 0:
            print(f"  Label {label}: {current_count} samples >= {samples_per_class}, skip")
            continue

        print(f"  Label {label}: {current_count} existing, generating {n_to_generate}...", end=" ", flush=True)

        metadata = Metadata.detect_from_dataframe(data=df_label, table_name=str(label))
        synthesizer = TVAESynthesizer(
            metadata,
            embedding_dim=64,
            compress_dims=[128, 64],
            decompress_dims=[64, 128],
            l2scale=1e-4,
            loss_factor=2.0,
            batch_size=args.batch_size,
            epochs=args.epochs,
            cuda=use_cuda,
        )
        synthesizer.fit(df_label)
        df_synth = synthesizer.sample(num_rows=n_to_generate)
        df_synth["Label"] = label
        synthetic_dfs.append(df_synth)
        print("done")

    if not synthetic_dfs:
        print("Nothing to augment.")
        return

    synthetic_all = pd.concat(synthetic_dfs, axis=0, ignore_index=True)
    final_df = pd.concat([train_df, synthetic_all], axis=0, ignore_index=True)

    # Remove duplicates
    before = len(final_df)
    dup_mask = final_df.drop(columns='Label').duplicated(keep='first')
    final_df = final_df[~dup_mask].reset_index(drop=True)
    print(f"Removed {before - len(final_df)} duplicates")

    # -- Save --
    out_dir = get_path('augment_dir')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f'train_tvae_{args.total_samples}.csv')
    final_df.to_csv(out_path, index=False)

    synth_path = os.path.join(out_dir, f'tvae_synthetic_{args.total_samples}.csv')
    synthetic_all.to_csv(synth_path, index=False)

    print(f"\nSaved augmented:  {out_path} ({len(final_df)} rows)")
    print(f"Saved synthetic:  {synth_path} ({len(synthetic_all)} rows)")

    # -- Label distribution --
    print(f"\nLabel distribution:")
    for label in all_labels:
        count = len(final_df[final_df['Label'] == label])
        print(f"  {label}: {count}")

    # -- Quick XGB eval --
    if not args.no_eval:
        print("\nXGB evaluation...")
        test_df = pd.read_csv(get_path('test'), low_memory=False)
        X_aug = final_df.drop('Label', axis=1).values
        y_aug = final_df['Label'].values
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_aug, y_aug, test_size=0.1, random_state=42, stratify=y_aug,
        )

        xgb = XGBClassifier(
            tree_method='auto', learning_rate=0.4, eval_metric='auc',
            objective='multi:softprob', random_state=42, early_stopping_rounds=20,
            verbosity=0,
        )
        xgb.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
        y_pred = xgb.predict(X_val)

        print(f"  Accuracy: {sklearn.metrics.accuracy_score(y_val, y_pred):.4f}")
        print(f"  Macro-F1: {sklearn.metrics.f1_score(y_val, y_pred, average='macro'):.4f}")
        print(f"  Micro-F1: {sklearn.metrics.f1_score(y_val, y_pred, average='micro'):.4f}")


if __name__ == '__main__':
    main()
