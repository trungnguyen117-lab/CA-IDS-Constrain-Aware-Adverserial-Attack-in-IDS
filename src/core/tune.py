"""Tuning logic: DE search for static ensemble weights (no CLI).

mi4 không tune — chỉ ghi default ``MI_DEFAULTS`` vào yaml params; tại
runtime, ``MutualInferenceV4`` tự tính trọng số động per-sample, dùng
``w_static`` từ ``ensemble_static.yaml`` làm tree-base.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import yaml
from scipy.optimize import differential_evolution
from sklearn.metrics import f1_score

from .data import load_dataset, load_models
from .ensemble import MI_DEFAULTS
from .predict import build_art_predictors

logger = logging.getLogger(__name__)


def softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def macro_f1(y_true, P) -> float:
    return f1_score(y_true, P.argmax(axis=1), average="macro", zero_division=0)


def gather_clean_proba(cfg, models, X, device):
    """Return ``{target: (N, C) softmax probabilities}``."""
    predictors = build_art_predictors(cfg, models, X, device=device)
    return {n: predictors[n](X) for n in models}


def run_de(neg_obj, bounds, *, seed, maxiter, popsize):
    res = differential_evolution(
        neg_obj, bounds=bounds,
        seed=seed, maxiter=maxiter, popsize=popsize, tol=1e-9,
        mutation=(0.5, 1.5), recombination=0.9,
    )
    return res.x, float(-res.fun)


# ── Step 1: static weights ────────────────────────────────────────────────────


def tune_static(cfg, P_clean, y, *, seed, maxiter, popsize):
    names = list(cfg.all_targets)
    n = len(names)

    def neg_f1(z):
        w = softmax(np.asarray(z, dtype=float))
        ens = sum(w[i] * P_clean[names[i]] for i in range(n))
        return -macro_f1(y, ens)

    x, f1 = run_de(neg_f1, [(-3.0, 3.0)] * n,
                   seed=seed, maxiter=maxiter, popsize=popsize)
    w = softmax(np.asarray(x, dtype=float))
    return {names[i]: float(round(w[i], 4)) for i in range(n)}, f1


# ── Output ────────────────────────────────────────────────────────────────────


def write_ensemble_yaml(path: Path, *, weights: dict | None = None,
                        params: dict | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    body: dict = {}
    if weights is not None:
        body["weights"] = dict(weights)
    if params is not None:
        body["params"] = dict(params)
    with open(path, "w") as f:
        yaml.safe_dump(body, f, sort_keys=False)
    logger.info("→ %s", path)


# ── End-to-end tuning ─────────────────────────────────────────────────────────


def run_tune_ensemble(cfg, *, at, val_path, model_dir, defense_model_dir,
                      output_dir, de_iter, de_popsize, seed, device):
    """Tune static weights, ghi yaml cho static / mi4."""
    de_kwargs = {"seed": seed, "maxiter": de_iter, "popsize": de_popsize}

    df, X, y, _ = load_dataset(cfg, val_path or "test")
    logger.info("Val data: (%s)", df.shape)

    models = load_models(
        cfg, cfg.all_targets,
        defense=("at" if at else None), device=device,
        model_dir=str(model_dir) if model_dir else None,
        defense_model_dir=str(defense_model_dir) if defense_model_dir else None,
    )
    if not models:
        sys.exit("No models loaded — check --model-dir / --defense-model-dir")
    logger.info("Loaded %d models (%s): %s",
                len(models), "AT" if at else "baseline", list(models))

    P = gather_clean_proba(cfg, models, X, device)

    logger.info("=" * 60)
    logger.info("Step 1 — DE tune static weights (%d-D simplex)", len(cfg.all_targets))
    w_static, f1_static = tune_static(cfg, P, y, **de_kwargs)
    logger.info("static F1=%.4f  weights=%s", f1_static, w_static)

    from .mutual_inference_v5 import MI5_DEFAULTS

    out_dir = output_dir or (cfg.root / "config")
    write_ensemble_yaml(out_dir / "ensemble_static.yaml", weights=w_static)
    write_ensemble_yaml(out_dir / "ensemble_mi4.yaml", params=dict(MI_DEFAULTS))
    write_ensemble_yaml(out_dir / "ensemble_mi5.yaml", params=dict(MI5_DEFAULTS))

    print("\n" + "=" * 60)
    print(f" Ensemble tune summary  ({'AT' if at else 'baseline'} models)")
    print("=" * 60)
    print(f"   static  Macro-F1 = {f1_static:.4f}")
    print(f"   weights = {w_static}")
    print(f"   mi4 params = {dict(MI_DEFAULTS)} (defaults, no tuning)")
    print(f"   mi5 params = {dict(MI5_DEFAULTS)} (defaults, no tuning)")
    print(f"   wrote yaml → {out_dir}/ensemble_*.yaml")
