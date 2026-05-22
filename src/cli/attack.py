"""Stage 2 — Adversarial sample generation."""

from __future__ import annotations

import argparse
import logging
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

from ..core.attack_runner import run_task
from ..core.config import Config
from ..core.data import load_dataset
from ..core.metrics import get_mutate_indices
from ..core.paths import resolve_arg
from .runtime import add_io_args, add_runtime_args

logger = logging.getLogger(__name__)


def generate(cfg: Config, args: argparse.Namespace) -> None:
    targets = list(cfg.all_targets) if "all" in args.target else args.target
    attacks = (list(cfg.wb_attacks) + list(cfg.bb_attacks)
               if "all" in args.attack else args.attack)

    data_key = args.train_path if args.source == "train" and args.train_path else (
        args.test_path if args.source == "test" and args.test_path else args.source
    )
    df, X, y, feats = load_dataset(cfg, data_key, max_samples=args.max_samples)
    protected_idx = get_mutate_indices(df, label_col=cfg.label_col,
                                       cont_features=cfg.cont_features or None)
    protected_idx = []
    logger.info("Data: %s | Protected: %d", X.shape, len(protected_idx))
    _, X_ref, _, _ = load_dataset(cfg, data_key)
    logger.info("Clip/eps reference: train %s", X_ref.shape)

    model_dir = resolve_arg(cfg, args.model_dir)
    adv_dir = resolve_arg(cfg, args.adv_dir)
    tasks = []
    for t in targets:
        for a in attacks:
            if a not in cfg.attack_compat.get(t, []):
                logger.warning("Skip %s/%s (incompatible)", t, a)
                continue
            gen_kwargs = None
            if a in cfg.bb_attacks:
                gen_kwargs = {
                    "batch_size": args.bb_batch_size,
                    "timeout": args.bb_timeout,
                    "max_retries": args.bb_max_retries,
                    "placeholder": args.bb_placeholder,
                    "verbose": args.bb_verbose,
                }
            tasks.append((
                cfg, t, a, args.source, args.device,
                X, y, feats, protected_idx, args.pass_y,
                model_dir, str(adv_dir) if adv_dir else None,
                args.log_level, gen_kwargs, X_ref,
            ))

    logger.info("Tasks: %d | Workers: %d", len(tasks), args.workers)
    if args.workers <= 1:
        for task in tasks:
            _, _, err = run_task(task)
            if err:
                logger.error("Failed:\n%s", err)
        return
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=min(args.workers, len(tasks)), mp_context=ctx) as pool:
        futures = {pool.submit(run_task, t): (t[1], t[2]) for t in tasks}
        for fut in as_completed(futures):
            tgt, atk = futures[fut]
            _, _, err = fut.result()
            if err:
                logger.error("Failed %s/%s:\n%s", tgt, atk, err)
            else:
                logger.info("Done: %s/%s", tgt, atk)


def register(sub: argparse._SubParsersAction, cfg: Config) -> None:
    targets = list(cfg.all_targets) + list(cfg.surrogate_targets) + ["all"]
    attacks = list(cfg.wb_attacks) + list(cfg.bb_attacks) + ["all"]
    p = sub.add_parser("gen-adv", help="generate adversarial CSVs (per target × attack)")
    p.add_argument("--target", "-t", nargs="+", required=True, choices=targets)
    p.add_argument("--attack", "-a", nargs="+", required=True, choices=attacks)
    p.add_argument("--source", "-s", required=True, choices=["train", "test"])
    p.add_argument("--max-samples", "-n", type=int, default=None)
    p.add_argument("--bb-batch-size", type=int, default=-1)
    p.add_argument("--bb-timeout", type=int, default=-1)
    p.add_argument("--bb-max-retries", type=int, default=3)
    p.add_argument("--bb-placeholder", default="original",
                   choices=["original", "drop"])
    p.add_argument("--bb-verbose", type=int, default=10)
    p.add_argument("--pass-y", action="store_true")
    p.add_argument("--workers", "-w", type=int, default=1)
    add_io_args(p, train=True, test=True, model=True, adv=True)
    add_runtime_args(p)
    p.set_defaults(func=generate)
