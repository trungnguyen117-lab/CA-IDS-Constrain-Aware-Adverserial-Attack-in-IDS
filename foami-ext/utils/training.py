"""Training utilities: ModelManager, ART classifier building, checkpoint saving."""
import os
import logging

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split

from .config import load_training_config
from .constants import MODEL_FILENAMES, LOG_TAG
from .evaluation import report_metrics

logger = logging.getLogger(__name__)


class ModelManager:
    """Unified training manager for all model types (tree + DL).

    Usage:
        from utils.data import DataManager
        dm = DataManager(train_csv, test_csv)
        mm = ModelManager(dm, models_dir='models/', device='cpu')
        mm.train('xgb')
        mm.train('lstm')
        mm.train_all()  # train all 5 models
    """

    def __init__(self, dm, models_dir, device='cpu'):
        from model import MODEL_REGISTRY
        self._dm = dm
        self._models_dir = models_dir
        self._device = device
        self._registry = MODEL_REGISTRY
        os.makedirs(models_dir, exist_ok=True)

    def train(self, model_name, out_name=None):
        """Train a single model by name. Dispatches to tree or DL logic."""
        if model_name not in self._registry:
            raise ValueError(f"Unknown model: {model_name}. "
                             f"Available: {list(self._registry)}")

        ModelClass, model_type = self._registry[model_name]
        out_name = out_name or MODEL_FILENAMES[model_name]
        tag = LOG_TAG[model_name]

        X_train, y_train = self._dm.train_data
        X_test, y_test = self._dm.test_data
        num_class = self._dm.num_classes

        logger.info(f"[{tag}] Starting training ...")

        if model_type == 'tree':
            self._train_tree(model_name, ModelClass, X_train, y_train,
                             X_test, y_test, num_class, tag, out_name)
        else:
            input_dim = self._dm.input_dim
            self._train_dl(model_name, ModelClass, X_train, y_train,
                           X_test, y_test, input_dim, num_class, tag, out_name)

    def train_multiple(self, model_names, out_names=None):
        """Train multiple models sequentially."""
        out_names = out_names or {}
        for m in model_names:
            self.train(m, out_name=out_names.get(m))

    def train_all(self, out_names=None):
        """Train all registered models."""
        self.train_multiple(list(self._registry), out_names)

    def _train_tree(self, model_name, ModelClass, X_train, y_train,
                    X_test, y_test, num_class, tag, out_name):
        cfg = load_training_config(model_name)

        if model_name == 'xgb':
            cfg['device'] = 'cuda' if self._device in ('cuda', 'gpu') else 'cpu'
        elif model_name == 'cat':
            cfg['task_type'] = 'GPU' if self._device in ('cuda', 'gpu') else 'CPU'
            cfg['classes_count'] = num_class

        random_state = cfg.pop('random_state', 42)
        model = ModelClass(num_class=num_class, params=cfg, random_state=random_state)
        model.fit(X_train, y_train)

        logger.info(f"[{tag}] Training complete")
        report_metrics(tag, y_test, model.predict(X_test))
        model.save_model(os.path.join(self._models_dir, out_name))

    def _train_dl(self, model_name, ModelClass, X_train, y_train,
                  X_test, y_test, input_dim, num_class, tag, out_name):
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_train, y_train, test_size=0.1, random_state=42, stratify=y_train
        )
        cfg = load_training_config(model_name)
        model = ModelClass(input_dim=input_dim, num_class=num_class,
                           device=self._device, **cfg)
        model.fit(X_tr, y_tr, X_val, y_val)

        logger.info(f"[{tag}] Training complete")
        report_metrics(tag, y_test, model.predict(X_test))
        model.save_model(os.path.join(self._models_dir, out_name))

_ARCH_KEYS = {
    'lstm':   ('step_dim', 'hidden', 'layers', 'n_classes', 'dropout', 'bidir'),
    'resdnn': ('in_dim', 'n_classes'),
}


def build_art_classifier(model, input_dim, num_classes, clip_values,
                         device_type, lr=1e-3, weight_decay=2e-4,
                         preprocessing=None):
    """Wrap PyTorch model in ART PyTorchClassifier with AdamW.

    The optimizer is attached at construction time so that AdversarialTrainer
    can invoke classifier.fit() (which calls optimizer.step()) internally.

    preprocessing: (mean, std) arrays from the wrapper's scaler — ART applies
    (x - mean) / std before forward pass, so attacks operate in raw feature space.
    """
    from art.estimators.classification import PyTorchClassifier

    optimizer = optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )
    return PyTorchClassifier(
        model=model,
        loss=nn.CrossEntropyLoss(),
        optimizer=optimizer,
        input_shape=(input_dim,),
        nb_classes=num_classes,
        clip_values=clip_values,
        preprocessing=preprocessing,
        device_type=device_type,
    )


def save_dl_checkpoint(classifier, model_name, original_ckpt_path,
                       out_path, scaler_mean, scaler_scale):
    """Save AT-updated weights, unified for lstm/resdnn.

    Reads architecture hyperparams from the original checkpoint,
    combines with updated state_dict and scaler params.
    """
    if model_name not in _ARCH_KEYS:
        raise ValueError(f"Unknown DL model: {model_name}. Supported: {list(_ARCH_KEYS)}")

    orig = torch.load(original_ckpt_path, map_location='cpu', weights_only=False)
    ckpt = {
        'state_dict': classifier.model.state_dict(),
        'scaler_mean': scaler_mean,
        'scaler_scale': scaler_scale,
    }
    for key in _ARCH_KEYS[model_name]:
        ckpt[key] = orig[key]
    torch.save(ckpt, out_path)
    logger.info(f"[{model_name.upper()}-AT] Saved → {out_path}")
