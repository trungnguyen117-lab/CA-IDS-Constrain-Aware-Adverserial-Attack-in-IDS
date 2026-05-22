"""Shared CLI helpers: runtime flags, I/O flag presets, format parsing."""

from __future__ import annotations

import argparse

from ..core.paths import resolve_device, setup_logging


def add_runtime_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--device", "-d", default="cpu",
                   choices=["cpu", "cuda", "auto", "mps"])
    p.add_argument("--log-level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])


_IO_FLAGS = {
    "train": "--train-path",
    "test": "--test-path",
    "out": "--out-dir",
    "model": "--model-dir",
    "defense_model": "--defense-model-dir",
    "adv": "--adv-dir",
    "ensemble_cfg": "--ensemble-config-dir",
}


def add_io_args(p: argparse.ArgumentParser, **flags: bool) -> None:
    for key, enabled in flags.items():
        if enabled:
            p.add_argument(_IO_FLAGS[key], default=None)


def init_runtime(args: argparse.Namespace) -> argparse.Namespace:
    setup_logging(args.log_level)
    args.device = resolve_device(args.device)
    return args


def parse_formats(spec: str) -> tuple[str, ...]:
    parts = tuple(s.strip() for s in (spec or "").split(",") if s.strip())
    valid = {"md", "txt"}
    bad = [p for p in parts if p not in valid]
    if bad:
        raise SystemExit(f"Unknown --export-format value(s): {bad}; valid: {sorted(valid)}")
    return parts or ("md", "txt")
