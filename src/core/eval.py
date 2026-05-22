"""Evaluation logic: single-model + ensemble + scenario evaluators (no CLI).

- ``Evaluator`` / ``EvalConfig``: clean + adv eval for one model.
- ``ensemble_eval(...)``: clean + adv eval for one (strategy, defense) ensemble.
- ``run_scenarios(...)``: orchestrate single + ensemble × 4 scenarios.
- ``load_tuned_config(...)``: read ``ensemble_<strategy>.yaml``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from .predict import build_art_predictor, build_art_predictors
from .data import load_model, load_models
from .ensemble import Ensemble
from .metrics import (
    compute_asr, find_adv_csvs, load_adv_features, report_metrics,
)

logger = logging.getLogger(__name__)


# ── Single-model eval ────────────────────────────────────────────────────────


@dataclass
class EvalConfig:
    device: str = "cpu"
    defense: str | None = None
    model_dir: str | None = None
    defense_model_dir: str | None = None
    adv_dir: str | None = None
    preprocessing_defence: object = None


class Evaluator:
    def __init__(self, cfg, config: EvalConfig | None = None):
        self.cfg = cfg
        self.config = config or EvalConfig()

    def evaluate_single(self, target, X_test, y_test, feature_names,
                        adv_files: dict | None = None) -> dict:
        cfg = self.config
        tag = f"{target.upper()} ({cfg.defense or 'baseline'})"

        model = load_model(self.cfg, target, defense=cfg.defense, device=cfg.device,
                           model_dir=cfg.model_dir,
                           defense_model_dir=cfg.defense_model_dir)
        preproc = (cfg.preprocessing_defence or {}).get(target)
        predict_proba = build_art_predictor(
            self.cfg, model, target, X_test, device=cfg.device,
            preprocessing_defences=preproc,
        )

        def predict_fn(X):
            return np.argmax(predict_proba(X), axis=1)

        y_clean = predict_fn(X_test)
        results = {"clean": report_metrics(f"{tag} clean", y_test, y_clean)}

        if adv_files is None:
            adv_dir = cfg.adv_dir or str(self.cfg.adv_eval_path())
            adv_files = find_adv_csvs(target, adv_dir, cfg=self.cfg)
        for atk, path in (adv_files or {}).items():
            m = self.eval_adv(path, predict_fn, y_test, y_clean, feature_names,
                              tag=f"{tag} {atk}")
            if m:
                results[atk] = m
        return results

    def eval_adv(self, csv_path, predict_fn, y_test, y_clean, feature_names,
                 tag=""):
        X_adv = load_adv_features(csv_path, feature_names)
        if X_adv is None:
            return None
        y_adv = predict_fn(X_adv)
        m = report_metrics(tag, y_test, y_adv)
        m["asr"] = compute_asr(y_test, y_clean, y_adv)
        return m


# ── Ensemble eval ────────────────────────────────────────────────────────────


def _read_yaml(bases, name):
    for base in bases:
        path = base / name
        if path.is_file():
            with open(path) as f:
                return yaml.safe_load(f) or {}, path
    return None, None


def load_tuned_config(cfg, strategy: str, config_dir=None):
    """Resolve ensemble config.

    - ``static``: weights từ ``ensemble_static.yaml``.
    - ``mi4``: weights lấy từ ``ensemble_static.yaml``; params lấy từ
      ``ensemble_mi4.yaml`` (optional, fallback default).
    """
    bases = []
    if config_dir is not None:
        bases.append(Path(config_dir))
    bases.append(cfg.root / "config")

    if strategy == "static":
        data, _ = _read_yaml(bases, "ensemble_static.yaml")
        if data is None:
            return None, None
        return data.get("weights"), data.get("params")

    static_data, static_path = _read_yaml(bases, "ensemble_static.yaml")
    if static_data is None:
        raise FileNotFoundError(
            f"ensemble_static.yaml not found in {bases}; "
            f"strategy={strategy!r} requires it as the weights source."
        )
    weights = static_data.get("weights")

    strat_data, _ = _read_yaml(bases, f"ensemble_{strategy}.yaml")
    params = (strat_data or {}).get("params")
    logger.info("Loaded weights from %s, params from ensemble_%s.yaml",
                static_path, strategy)
    return weights, params


def load_adv_per_target(target_maps, attack, names, feats, expected_n):
    """Load adv CSVs for ``attack`` across all targets. Skip if any missing."""
    per = {}
    for n in names:
        path = target_maps[n].get(attack)
        if path is None:
            return None
        X_adv = load_adv_features(path, feats, expected_n=expected_n)
        if X_adv is None:
            return None
        per[n] = X_adv
    return per


def ensemble_eval(cfg, strategy, at, device, model_dir, defense_model_dir, adv_dir,
                  X_test, y_test, feats, config_dir=None,
                  defense="at", preprocessing_defence=None) -> dict:
    """Evaluate one ensemble (strategy, at) on clean + adv attacks."""
    models = load_models(cfg, cfg.all_targets,
                         defense=(defense if at else None), device=device,
                         model_dir=model_dir, defense_model_dir=defense_model_dir)
    if not models:
        logger.warning("No models loaded")
        return {}
    proba = build_art_predictors(cfg, models, X_test, device=device,
                                 preprocessing_defences=preprocessing_defence)

    weights, params = load_tuned_config(cfg, strategy, config_dir=config_dir)
    ens = Ensemble(cfg, strategy=strategy, weights=weights, params=params)
    clean_p = {n: proba[n](X_test) for n in models}
    _, y_clean = ens.predict(clean_p)
    tag = f"{strategy.upper()} ENS ({defense.upper() if at else 'baseline'})"
    results = {"clean": report_metrics(f"{tag} clean", y_test, y_clean)}

    adv_root = adv_dir or str(cfg.adv_eval_path())
    target_maps = {n: find_adv_csvs(n, adv_root, cfg=cfg) for n in models}
    all_attacks = sorted({a for m in target_maps.values() for a in m})
    N = len(X_test)
    names = list(models)

    for atk in all_attacks:
        per = load_adv_per_target(target_maps, atk, names, feats, N)
        if per is None:
            continue
        adv_p = {n: proba[n](per[n]) for n in names}
        _, y_adv = ens.predict(adv_p)
        m = report_metrics(f"{tag} {atk}", y_test, y_adv)
        m["asr"] = compute_asr(y_test, y_clean, y_adv)
        results[atk] = m
    return results


# ── Scenario eval (S1, S2, E1, E2) ───────────────────────────────────────────


SCEN_META = {
    "S1": dict(exp="S1_baseline_single", desc="Baseline single — no defense", scope="single",   at=False),
    "S2": dict(exp="S2_at_single",       desc="AT single",                    scope="single",   at=True),
    "E1": dict(exp="E1_baseline_ens",    desc="Baseline ensemble",            scope="ensemble", at=False),
    "E2": dict(exp="E2_at_ens",          desc="AT ensemble",                  scope="ensemble", at=True),
}

SINGLE_SCENARIOS = {"S1": False, "S2": True}    # value = at
ENS_SCENARIOS    = {"E1": False, "E2": True}


def exp_id(scen, key, scope, label=None):
    """Build exp_id like 'S1_baseline_single__cat' or 'E2_at_ens_static'."""
    base = label if label else SCEN_META[scen]["exp"]
    return f"{base}__{key}" if scope == "single" else f"{base}_{key}"


def run_scenarios(cfg, *, targets, strategies, scenarios,
                  X_te, y_te, feats,
                  device="cpu",
                  model_dir=None, defense_model_dir=None, adv_dir=None,
                  ensemble_config_dir=None,
                  defense="at", preprocessing_defence=None) -> list[tuple]:
    """Run S1/S2/E1/E2 scenarios. Returns ``[(scenario, target_or_strategy, results), ...]``."""
    rows: list[tuple] = []
    for s in scenarios:
        if s in SINGLE_SCENARIOS:
            at = SINGLE_SCENARIOS[s]
            ev_cfg = EvalConfig(
                device=device, defense=(defense if at else None),
                model_dir=model_dir, defense_model_dir=defense_model_dir,
                adv_dir=adv_dir,
                preprocessing_defence=preprocessing_defence,
            )
            ev = Evaluator(cfg, ev_cfg)
            for t in targets:
                rows.append((s, t, ev.evaluate_single(t, X_te, y_te, feats)))
        elif s in ENS_SCENARIOS:
            at = ENS_SCENARIOS[s]
            for strat in strategies:
                res = ensemble_eval(cfg, strat, at, device,
                                    model_dir, defense_model_dir, adv_dir,
                                    X_te, y_te, feats,
                                    config_dir=ensemble_config_dir,
                                    defense=defense,
                                    preprocessing_defence=preprocessing_defence)
                rows.append((s, strat, res))
    return rows
