"""Stage 4 — Tune ensemble weights & MI hyperparams (DE search)."""

from __future__ import annotations

import argparse

from ..core.config import Config
from ..core.paths import resolve_arg
from ..core.tune import run_tune_ensemble
from .runtime import add_io_args, add_runtime_args


def tune_ensemble(cfg: Config, args: argparse.Namespace) -> None:
    run_tune_ensemble(
        cfg,
        at=(args.at == "true"),
        val_path=args.val_path,
        model_dir=resolve_arg(cfg, args.model_dir),
        defense_model_dir=resolve_arg(cfg, args.defense_model_dir),
        output_dir=resolve_arg(cfg, args.output_dir, cfg.root / "config"),
        de_iter=args.de_iter if args.de_iter is not None else cfg.de_iter,
        de_popsize=args.de_popsize if args.de_popsize is not None else cfg.de_popsize,
        seed=args.seed if args.seed is not None else cfg.tune_seed,
        device=args.device,
    )


def register(sub: argparse._SubParsersAction, cfg: Config) -> None:
    p = sub.add_parser("tune-ensemble",
                       help="tune ensemble weights & MI hyperparams via DE")
    p.add_argument("--at", default="false", choices=["false", "true"])
    p.add_argument("--val-path", default=None)
    p.add_argument("--output-dir", default=None,
                   help="dir to write ensemble_<strategy>{_at}.yaml (default: <root>/config)")
    p.add_argument("--de-iter", type=int, default=None)
    p.add_argument("--de-popsize", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    add_io_args(p, model=True, defense_model=True)
    add_runtime_args(p)
    p.set_defaults(func=tune_ensemble)
