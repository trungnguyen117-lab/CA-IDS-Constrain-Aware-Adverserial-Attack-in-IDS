"""Adversarial training for DL models (LSTM, ResDNN).

Loads a pre-trained checkpoint, wraps it in an ART PyTorchClassifier,
creates an attack generator from src/art_generator/, then runs
foami+/defense/AdversarialTrainingWrapper to fine-tune the model.

The default training data is the TVAE-augmented dataset produced by
prepare_adv_data.py (adv_samples/adv_training/train_at.csv).

Pipeline:
    DS/train_shap_66.csv
        ↓  prepare_adv_data.py   (TVAE augmentation)
    adv_samples/adv_training/train_at.csv
        ↓  adv_train_dl.py       (this script)
    models/framework_{model}_TVAE_at_{attack}.pth

The StandardScaler is baked into _ScaledLSTM / _ScaledResDNN so ART
operates in raw-feature space — consistent with generate_adv_soict.py.

Output checkpoints are compatible with:
    LSTMWrapper.from_checkpoint()   (src/art_classifier/lstm_classifier.py)
    ResDNNWrapper.from_checkpoint() (src/art_classifier/resdnn_classifier.py)

Supported attacks: pgd, fgsm
Params loaded from foami+/config/adv_training/{attack}.yaml — all overridable.

Usage:
    # Step 1 — prepare augmented data (run once)
    python prepare_adv_data.py

    # Step 2 — adversarial training
    python adv_train_dl.py --model lstm   --attack pgd
    python adv_train_dl.py --model resdnn --attack fgsm --epochs 50 --eps 0.05
    python adv_train_dl.py --model lstm resdnn --attack pgd --device cuda

    # Use original (non-augmented) training data instead
    python adv_train_dl.py --model lstm --attack pgd --train-csv ../../DS/train_tvae.csv
"""

import os
import sys
import argparse

# ── macOS / PyTorch compatibility (must be set before torch is imported) ───────
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')

import numpy as np
import pandas as pd

# ── Path bootstrap ──────────────────────────────────────────────────────────────
_HERE  = os.path.dirname(os.path.realpath(__file__))
_FOAMI = os.path.dirname(os.path.dirname(_HERE))   # foami+/
sys.path.insert(0, _FOAMI)

from utils.paths import setup_paths, MODELS_DIR, TEST_CSV, AT_TRAIN_CSV
setup_paths()   # adds SRC_DIR (art_classifier, art_generator) and FOAMI_DIR

from utils.logging          import setup_logging, get_logger
from utils.evaluation       import report_metrics
from utils.config           import load_adv_training_config
from utils.loaders          import load_wrapper
from defense.adversarial_training import AdversarialTrainingWrapper

import torch
import torch.nn as nn
import torch.optim as optim
torch.set_num_threads(1)   # prevent OMP thread-pool conflicts on macOS / CPU

logger = get_logger(__name__)

DL_MODELS   = ['lstm', 'resdnn']
AT_ATTACKS  = ['pgd', 'fgsm']
_ALL_CHOICE = 'all'


# ── ART classifier builder ─────────────────────────────────────────────────────

def _build_art_classifier(scaled_model, input_dim: int, num_classes: int,
                           clip_values: tuple, device_type: str,
                           lr: float, weight_decay: float):
    """Wrap a _ScaledLSTM/_ScaledResDNN in ART PyTorchClassifier with AdamW.

    The optimizer is attached at construction time so that AdversarialTrainer
    can invoke classifier.fit() (which calls optimizer.step()) internally.
    """
    from art.estimators.classification import PyTorchClassifier

    optimizer = optim.AdamW(
        scaled_model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )
    return PyTorchClassifier(
        model=scaled_model,
        loss=nn.CrossEntropyLoss(),
        optimizer=optimizer,
        input_shape=(input_dim,),
        nb_classes=num_classes,
        clip_values=clip_values,
        device_type=device_type,
    )


# ── Attack generator builder ───────────────────────────────────────────────────

def _make_generator(attack_name: str, classifier, cfg: dict):
    """Create a non-targeted attack generator from src/art_generator/.

    Keys that belong to the training loop (not the generator) are excluded.
    targeted is always forced to False — required by ART AdversarialTrainer.
    """
    from art_generator.pgd  import PGDAttackGenerator
    from art_generator.fgsm import FGSMAttackGenerator

    _LOOP_KEYS = {'epochs', 'ratio', 'lr', 'weight_decay'}
    generator_params = {k: v for k, v in cfg.items() if k not in _LOOP_KEYS}
    generator_params['targeted'] = False   # AdversarialTrainer requires this

    if attack_name == 'pgd':
        return PGDAttackGenerator(classifier, generator_params=generator_params)
    if attack_name == 'fgsm':
        return FGSMAttackGenerator(classifier, generator_params=generator_params)
    raise ValueError(f"Unsupported AT attack: {attack_name!r}. Choose: {AT_ATTACKS}")


# ── Checkpoint savers ──────────────────────────────────────────────────────────

def _save_lstm_checkpoint(classifier, original_ckpt_path: str, out_path: str) -> None:
    """Save updated LSTM weights compatible with LSTMWrapper.from_checkpoint().

    classifier.model → ART .model property → _ScaledLSTM (updated in-place)
    _ScaledLSTM.lstm → LSTMTabular whose state_dict() we save as 'state_dict'.
    Architecture hyperparams are re-read from the original checkpoint.
    """
    orig         = torch.load(original_ckpt_path, map_location='cpu', weights_only=False)
    scaled_model = classifier.model   # _ScaledLSTM

    torch.save({
        'state_dict':   scaled_model.lstm.state_dict(),
        'step_dim':     orig['step_dim'],
        'hidden':       orig['hidden'],
        'layers':       orig['layers'],
        'n_classes':    orig['n_classes'],
        'dropout':      orig['dropout'],
        'bidir':        orig['bidir'],
        'scaler_mean':  scaled_model.mean_.cpu().numpy(),
        'scaler_scale': scaled_model.scale_.cpu().numpy(),
    }, out_path)
    logger.info(f"[LSTM-AT] Saved → {out_path}")


def _save_resdnn_checkpoint(classifier, original_ckpt_path: str, out_path: str) -> None:
    """Save updated ResDNN weights compatible with ResDNNWrapper.from_checkpoint()."""
    orig         = torch.load(original_ckpt_path, map_location='cpu', weights_only=False)
    scaled_model = classifier.model   # _ScaledResDNN

    torch.save({
        'state_dict':   scaled_model.resdnn.state_dict(),
        'in_dim':       orig['in_dim'],
        'n_classes':    orig['n_classes'],
        'scaler_mean':  scaled_model.mean_.cpu().numpy(),
        'scaler_scale': scaled_model.scale_.cpu().numpy(),
    }, out_path)
    logger.info(f"[ResDNN-AT] Saved → {out_path}")


# ── Per-model runner ───────────────────────────────────────────────────────────

def run_adv_training(model_name: str, attack_name: str,
                     X_train, y_train, X_test, y_test,
                     clip_values, num_classes: int, input_dim: int,
                     models_dir: str, device: str, cfg: dict) -> None:
    """Full adversarial training pipeline for one DL model."""

    ckpt_filename = f'framework_{model_name}_TVAE.pth'
    ckpt_path     = os.path.join(models_dir, ckpt_filename)
    if not os.path.exists(ckpt_path):
        raise SystemExit(
            f"[{model_name.upper()}-AT] Checkpoint not found: {ckpt_path}\n"
            "Run pipeline/0_training/train_dl.py first."
        )

    # ── 1. Load pre-trained model ──────────────────────────────────────────
    logger.info(f"[{model_name.upper()}-AT] Loading checkpoint: {ckpt_path}")
    wrapper      = load_wrapper(model_name, models_dir, clip_values,
                                num_classes, input_dim, device)
    scaled_model = wrapper.model   # _ScaledLSTM or _ScaledResDNN
    scaled_model.train()

    logger.info(f"[{model_name.upper()}-AT] Baseline on test set ...")
    report_metrics(f'{model_name.upper()}-baseline', y_test, wrapper.predict(X_test))

    # ── 2. Build ART PyTorchClassifier with AdamW ──────────────────────────
    device_type  = 'gpu' if device and 'cuda' in device else 'cpu'
    lr           = float(cfg.get('lr', 1e-3))
    weight_decay = float(cfg.get('weight_decay', 2e-4))

    logger.info(
        f"[{model_name.upper()}-AT] Building ART classifier: "
        f"input_dim={input_dim}, classes={num_classes}, device={device_type}, lr={lr}"
    )
    art_clf = _build_art_classifier(
        scaled_model, input_dim, num_classes, clip_values,
        device_type, lr, weight_decay,
    )

    # ── 3. Create attack generator (targeted=False enforced) ───────────────
    generator = _make_generator(attack_name, art_clf, cfg)
    logger.info(
        f"[{model_name.upper()}-AT] Attack: {type(generator).__name__}, "
        f"eps={getattr(generator.attack, 'eps', '?')}, targeted={generator.attack.targeted}"
    )

    # ── 4. Adversarial training via foami+/defense/ ────────────────────────
    nb_epochs  = int(cfg.get('epochs', 30))
    batch_size = int(cfg.get('batch_size', 128))
    ratio      = float(cfg.get('ratio', 0.5))

    at = AdversarialTrainingWrapper(art_clf, attack_generators=[generator])
    at.fit(X_train, y_train,
           batch_size=batch_size, nb_epochs=nb_epochs, ratio=ratio)

    # ── 5. Post-AT evaluation ──────────────────────────────────────────────
    logger.info(f"[{model_name.upper()}-AT] Post-AT evaluation ...")
    report_metrics(f'{model_name.upper()}-AT-{attack_name}', y_test, at.predict(X_test))

    # ── 6. Save adversarially-trained checkpoint ───────────────────────────
    out_path = os.path.join(models_dir, f'framework_{model_name}_TVAE_at_{attack_name}.pth')
    if model_name == 'lstm':
        _save_lstm_checkpoint(at.get_classifier(), ckpt_path, out_path)
    else:
        _save_resdnn_checkpoint(at.get_classifier(), ckpt_path, out_path)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Adversarial training for SOICT25 DL models"
    )
    parser.add_argument('--model', '-m', nargs='+', required=True,
                        choices=DL_MODELS + [_ALL_CHOICE],
                        help="Model(s) to train: lstm resdnn all")
    parser.add_argument('--attack', '-a', required=True, choices=AT_ATTACKS,
                        help="Attack for adversarial training: pgd fgsm")
    parser.add_argument('--train-csv', default=None,
                        help=f"Training CSV (default: {AT_TRAIN_CSV}). "
                             "Run prepare_adv_data.py first to generate it.")
    parser.add_argument('--test-csv', default=None,
                        help=f"Test CSV (default: {TEST_CSV})")
    parser.add_argument('--models-dir', default=None,
                        help=f"Model directory (default: {MODELS_DIR})")
    parser.add_argument('--device', '-d', default='cpu',
                        choices=['cpu', 'cuda', 'auto'])

    # YAML overrides — CLI values take priority over YAML
    parser.add_argument('--epochs',       type=int,   default=None)
    parser.add_argument('--batch-size',   type=int,   default=None)
    parser.add_argument('--ratio',        type=float, default=None,
                        help="Adversarial fraction per batch, e.g. 0.5")
    parser.add_argument('--eps',          type=float, default=None)
    parser.add_argument('--eps-step',     type=float, default=None, dest='eps_step')
    parser.add_argument('--max-iter',     type=int,   default=None, dest='max_iter')
    parser.add_argument('--lr',           type=float, default=None)
    parser.add_argument('--weight-decay', type=float, default=None, dest='weight_decay')
    parser.add_argument('--log-level', default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'])
    args = parser.parse_args()

    setup_logging(args.log_level)

    train_csv  = args.train_csv  or AT_TRAIN_CSV
    test_csv   = args.test_csv   or TEST_CSV
    models_dir = args.models_dir or MODELS_DIR

    if not os.path.exists(train_csv):
        raise SystemExit(
            f"Training data not found: {train_csv}\n"
            "Run prepare_adv_data.py first to generate the augmented dataset."
        )
    if not os.path.exists(test_csv):
        raise SystemExit(f"Test CSV not found: {test_csv}")
    os.makedirs(models_dir, exist_ok=True)

    logger.info(f"[+] Train : {train_csv}")
    logger.info(f"[+] Test  : {test_csv}")
    df_train = pd.read_csv(train_csv, low_memory=False)
    df_test  = pd.read_csv(test_csv,  low_memory=False)

    label_col = 'Label'
    feat_cols = [c for c in df_train.columns if c != label_col]
    X_train   = df_train[feat_cols].values.astype(np.float32)
    y_train   = df_train[label_col].values.astype(np.int64)
    X_test    = df_test[feat_cols].values.astype(np.float32)
    y_test    = df_test[label_col].values.astype(np.int64)

    num_classes = int(len(np.unique(y_train)))
    input_dim   = X_train.shape[1]
    clip_values = (float(X_train.min()), float(X_train.max()))

    logger.info(
        f"[+] Train={X_train.shape}, Test={X_test.shape}, "
        f"Classes={num_classes}, Features={input_dim}"
    )
    logger.info(f"[+] clip_values={clip_values}")

    # ── Merge YAML config with CLI overrides (CLI wins) ─────────────────────
    cfg = load_adv_training_config(args.attack)
    for key, val in {
        'epochs':       args.epochs,
        'batch_size':   args.batch_size,
        'ratio':        args.ratio,
        'eps':          args.eps,
        'eps_step':     args.eps_step,
        'max_iter':     args.max_iter,
        'lr':           args.lr,
        'weight_decay': args.weight_decay,
    }.items():
        if val is not None:
            cfg[key] = val

    logger.info(f"[+] AT config ({args.attack}): {cfg}")

    models = DL_MODELS if _ALL_CHOICE in args.model else args.model
    for m in models:
        run_adv_training(
            model_name=m,
            attack_name=args.attack,
            X_train=X_train, y_train=y_train,
            X_test=X_test,   y_test=y_test,
            clip_values=clip_values,
            num_classes=num_classes,
            input_dim=input_dim,
            models_dir=models_dir,
            device=args.device,
            cfg=cfg,
        )


if __name__ == '__main__':
    main()
