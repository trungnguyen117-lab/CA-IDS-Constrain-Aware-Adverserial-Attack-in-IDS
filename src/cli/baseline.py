"""Stage 1 — Baseline training."""

from __future__ import annotations

import argparse

from ..core.config import Config
from ..core.paths import resolve_arg
from ..core.train import run_baseline_training
from .runtime import add_io_args, add_runtime_args


def train_baseline(cfg: Config, args: argparse.Namespace) -> None:
    models = list(cfg.all_targets) if "all" in args.model else args.model
    out_dir = resolve_arg(cfg, args.out_dir)
    train_path = resolve_arg(cfg, args.train_path)
    test_path = resolve_arg(cfg, args.test_path)
    run_baseline_training(
        cfg, models, args.device, out_dir,
        train_path=str(train_path) if train_path else None,
        test_path=str(test_path) if test_path else None,
    )


def register(sub: argparse._SubParsersAction, cfg: Config) -> None:
    choices = list(cfg.all_targets) + list(cfg.surrogate_targets) + ["all"]
    p = sub.add_parser("train", help="train baseline models")
    p.add_argument("--model", "-m", nargs="+", required=True, choices=choices)
    add_io_args(p, train=True, test=True, out=True)
    add_runtime_args(p)
    p.set_defaults(func=train_baseline)
