"""Compose dataset-aware argparse from 5 pipeline stages."""

from __future__ import annotations

import argparse

from ..core.config import Config
from . import attack, baseline, defense, evaluation, tune


def build_parser(cfg: Config) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=f"aider {cfg.name}",
        description=f"Adversarial-robustness pipeline for {cfg.name}.",
    )
    sub = parser.add_subparsers(dest="command", required=True, metavar="<command>")
    for stage in (baseline, attack, defense, tune, evaluation):
        stage.register(sub, cfg)
    return parser
