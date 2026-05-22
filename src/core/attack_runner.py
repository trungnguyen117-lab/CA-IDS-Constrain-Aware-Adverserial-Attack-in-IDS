from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np

from .attacks import make_attack, wrap_for_art
from .data import load_model
from .paths import setup_logging

logger = logging.getLogger(__name__)


class Attacker:
    """Run one (target, attack) generation. Holds state for parallel workers."""

    def __init__(self, cfg, target: str, attack: str, device: str = "cpu",
                 model_dir: str | os.PathLike | None = None,
                 adv_dir: str | os.PathLike | None = None):
        self.cfg = cfg
        self.target = target
        self.attack = attack
        self.device = device
        self.model_dir = model_dir
        self.adv_dir = Path(adv_dir) if adv_dir else None
        self.is_dl = cfg.is_dl(target)

    def load_model(self):
        return load_model(self.cfg, self.target, device=self.device,
                          model_dir=self.model_dir)

    def wrap_art(self, model, X_ref):
        return wrap_for_art(self.cfg, model, self.target, X_ref,
                            device=self.device, attack=self.attack)


    def attack_cfg(self, source: str) -> dict:
        group = "adv_training" if source == "train" else "attacks"
        return self.cfg.cfg_yaml(group, self.attack)

    def run(self, X, y, feature_names, mutate_indices=(), source: str = "test",
            pass_y: bool = False, gen_kwargs: dict | None = None,
            X_ref: np.ndarray | None = None):
        ref = X if X_ref is None else X_ref
        model = self.load_model()
        art_clf = self.wrap_art(model, ref)
        params = self.attack_cfg(source)

        attack_generator = make_attack(self.attack, art_clf, params=params)
        meta = {"feature_names": feature_names, "label_column": self.cfg.label_col}
        kwargs = {"pass_y": pass_y, **(gen_kwargs or {})}
        return attack_generator.generate(X, y, meta, mutate_indices, **kwargs)

    def output_path(self, source: str) -> Path:
        return self.cfg.adv_csv(self.target, self.attack, source, base=self.adv_dir)

    def save(self, df_adv, source: str = "test") -> Path:
        path = self.output_path(source)
        path.parent.mkdir(parents=True, exist_ok=True)
        df_adv.to_csv(path, index=False)
        logger.info("Saved %d adv → %s", len(df_adv), path)
        return path

    def run_and_save(self, X, y, feature_names, mutate_indices=(),
                     source: str = "test", pass_y: bool = False,
                     gen_kwargs: dict | None = None,
                     X_ref: np.ndarray | None = None):
        df = self.run(X, y, feature_names, mutate_indices, source,
                      pass_y, gen_kwargs, X_ref=X_ref)
        return df, self.save(df, source)


def run_task(task):
    """Process-pool worker: unpacks tuple, instantiates Attacker, runs + saves."""
    (cfg, target, attack, source, device, X, y, feats, mutate_idx,
     pass_y, model_dir, adv_dir, log_level, gen_kwargs, X_ref) = task
    setup_logging(log_level)
    log = logging.getLogger(__name__)
    log.info("[pid=%d] %s | %s | %s", os.getpid(), target.upper(), attack.upper(), source)
    try:
        atk = Attacker(cfg, target, attack, device=device,
                       model_dir=model_dir, adv_dir=adv_dir)
        atk.run_and_save(X, y, feats, mutate_idx, source=source,
                         pass_y=pass_y, gen_kwargs=gen_kwargs, X_ref=X_ref)
        return target, attack, None
    except Exception:
        import traceback
        return target, attack, traceback.format_exc()
