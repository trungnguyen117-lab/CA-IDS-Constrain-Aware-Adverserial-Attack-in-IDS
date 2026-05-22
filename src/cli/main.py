"""Single CLI entry point for adversarial-robustness pipelines.

Usage::

    aider <dataset> <command> [flags...]
    aider <dataset> --help          # list subcommands
    aider <dataset> <command> -h    # flags for one subcommand

``dataset``  registry shortname (see ``datasets.yaml``) or path to ``config.yaml``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from ..core.config import Config
from .parser import build_parser
from .runtime import init_runtime


EXAMPLES = """\
examples:
  aider iiot train -m all --device cuda
  aider modbus eval-scenarios --scenarios S1 S2 --defense at
  aider iiot --help                       # list all subcommands
  aider iiot eval-scenarios --help        # flags for one subcommand
"""


def load_registry() -> dict[str, Path]:
    repo_root = Path(__file__).resolve().parents[2]
    reg_file = repo_root / "datasets.yaml"
    if not reg_file.is_file():
        return {}
    data = yaml.safe_load(reg_file.read_text()) or {}
    return {k: (repo_root / v).resolve() for k, v in data.get("datasets", {}).items()}


def resolve_config(arg: str, registry: dict[str, Path]) -> Path:
    if arg in registry:
        return registry[arg]
    p = Path(arg).resolve()
    if not p.is_file():
        raise SystemExit(
            f"Config not found: {arg!r}. "
            f"Use a path to config.yaml or a registry name: {sorted(registry)}"
        )
    return p


def main(argv: list[str] | None = None) -> None:
    registry = load_registry()

    bootstrap = argparse.ArgumentParser(
        prog="aider",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=EXAMPLES,
        add_help=True,
    )
    bootstrap.add_argument(
        "dataset",
        help=f"shortname {sorted(registry) or '(none in datasets.yaml)'} "
             "or path to config.yaml",
    )
    bootstrap.add_argument("rest", nargs=argparse.REMAINDER,
                           help="<command> [flags...]; use `aider <dataset> --help`")
    boot_args = bootstrap.parse_args(argv)

    cfg = Config.from_yaml(resolve_config(boot_args.dataset, registry))
    parser = build_parser(cfg)
    args = parser.parse_args(boot_args.rest)
    init_runtime(args)
    args.func(cfg, args)


if __name__ == "__main__":
    main()
