"""Stage 3 — Defense training (AT / PGD-AT / Distillation / MagNet)."""

from __future__ import annotations

import argparse
import logging
from functools import partial

import yaml

from ..core.config import Config
from ..core.data import load_dataset
from ..core.defenses import train_magnet_aes
from ..core.paths import resolve_arg
from ..core.train import load_attack_weights, train_adv_target, train_distill
from .runtime import add_io_args, add_runtime_args

logger = logging.getLogger(__name__)


def _run_adv(cfg: Config, args: argparse.Namespace, *,
             defense: str, attack_filter: list[str] | None = None) -> None:
    train_path = resolve_arg(cfg, args.train_path, cfg.paths.train)
    test_path = resolve_arg(cfg, args.test_path, cfg.paths.test)
    default_out = cfg.paths.models_pgd_at if defense == "pgd_at" else cfg.paths.models_at
    out_dir = resolve_arg(cfg, args.out_dir, default_out)
    adv_base = resolve_arg(cfg, args.adv_dir)
    targets = list(cfg.all_targets) if "all" in args.model else args.model

    extra: dict = {}
    if defense == "at":
        extra = dict(
            balance_adv=not args.no_balance_adv,
            per_attack_cap=args.per_attack_cap or None,
            adv_clean_ratio=args.adv_clean_ratio,
        )
    if attack_filter is not None:
        extra["attack_filter"] = attack_filter

    for t in targets:
        try:
            train_adv_target(
                cfg, t, str(train_path), str(test_path), str(out_dir),
                args.device, defense=defense,
                adv_base=str(adv_base) if adv_base else None,
                attack_weights=load_attack_weights(cfg, t) if defense == "at" else None,
                config_variant=args.config_variant,
                **extra,
            )
        except Exception:
            logger.exception("%s %s failed", defense.upper(), t)


def distill(cfg: Config, args: argparse.Namespace) -> None:
    out_dir = resolve_arg(cfg, args.out_dir, cfg.paths.models_distill)
    train_distill(
        cfg, args.target,
        train_path=args.train_path, test_path=args.test_path,
        out_dir=str(out_dir), device=args.device,
        temperature=args.temperature, epochs=args.epochs,
    )


def magnet(cfg: Config, args: argparse.Namespace) -> None:
    magnet_cfg_path = resolve_arg(
        cfg, args.config, cfg.root / "config" / "magnet.yaml",
    )
    with open(magnet_cfg_path) as f:
        magnet_yaml = yaml.safe_load(f)
    save_dir = resolve_arg(
        cfg, args.save_dir,
        cfg.root / magnet_yaml.get("save_dir", "defense/magnet"),
    )
    _, X, _, _ = load_dataset(cfg, args.train_path or "train")
    logger.info("Train: %s | save_dir: %s", X.shape, save_dir)
    train_magnet_aes(X, magnet_yaml, device=args.device, save_dir=str(save_dir))
    logger.info("MagNet AEs saved.")


def register(sub: argparse._SubParsersAction, cfg: Config) -> None:
    choices = list(cfg.all_targets) + list(cfg.surrogate_targets) + ["all"]

    # train-at
    p = sub.add_parser("train-at", help="adversarial training (AIDER multi-attack)")
    p.add_argument("--model", "-m", nargs="+", required=True, choices=choices)
    p.add_argument("--no-balance-adv", action="store_true")
    p.add_argument("--config-variant", default=None)
    p.add_argument("--per-attack-cap", type=int, default=None)
    p.add_argument("--adv-clean-ratio", type=float, default=1.0)
    add_io_args(p, train=True, test=True, out=True, adv=True)
    add_runtime_args(p)
    p.set_defaults(func=partial(_run_adv, defense="at"))

    # train-pgd
    p = sub.add_parser("train-pgd", help="PGD-only adversarial training (Madry baseline)")
    p.add_argument("--model", "-m", nargs="+", required=True, choices=choices)
    p.add_argument("--config-variant", default=None)
    add_io_args(p, train=True, test=True, out=True, adv=True)
    add_runtime_args(p)
    p.set_defaults(func=partial(_run_adv, defense="pgd_at", attack_filter=["pgd"]))

    # train-distill
    p = sub.add_parser("train-distill", help="defensive distillation (DL targets only)")
    default_dl = cfg.dl_targets[0] if cfg.dl_targets else None
    p.add_argument("--target", default=default_dl, choices=list(cfg.dl_targets))
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--temperature", "-T", type=float, default=20.0)
    add_io_args(p, train=True, test=True, out=True)
    add_runtime_args(p)
    p.set_defaults(func=distill)

    # train-magnet
    p = sub.add_parser("train-magnet", help="train MagNet autoencoders")
    p.add_argument("--config", default=None,
                   help="path to magnet.yaml (default: <root>/config/magnet.yaml)")
    p.add_argument("--save-dir", default=None)
    add_io_args(p, train=True)
    add_runtime_args(p)
    p.set_defaults(func=magnet)
