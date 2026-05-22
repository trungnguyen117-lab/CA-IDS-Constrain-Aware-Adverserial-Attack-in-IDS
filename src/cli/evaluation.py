"""Stage 5 — Evaluation (single / ensemble / scenarios)."""

from __future__ import annotations

import argparse

from ..core.config import Config
from ..core.data import load_dataset
from ..core.preprocessing_defenses import parse_fs_config
from ..core.eval import (
    EvalConfig, Evaluator, SCEN_META, ensemble_eval, exp_id, run_scenarios,
)
from ..core.export import write_results
from ..core.paths import resolve_arg
from .printers import print_compare, print_scenario_summary, print_summary
from .runtime import add_io_args, add_runtime_args, parse_formats


def single(cfg: Config, args: argparse.Namespace) -> None:
    targets = list(cfg.all_targets) if "all" in args.target else args.target
    _, X_te, y_te, feats = load_dataset(cfg, args.test_path or "test")

    def run(defense):
        ev_cfg = EvalConfig(
            device=args.device, defense=defense,
            model_dir=resolve_arg(cfg, args.model_dir, as_str=True),
            defense_model_dir=resolve_arg(cfg, args.defense_model_dir, as_str=True),
            adv_dir=resolve_arg(cfg, args.adv_dir, as_str=True),
        )
        ev = Evaluator(cfg, ev_cfg)
        return {t: ev.evaluate_single(t, X_te, y_te, feats) for t in targets}

    if args.at == "both":
        print_compare(run(None), run("at"))
    else:
        print_summary(run("at" if args.at == "true" else None))


def ensemble(cfg: Config, args: argparse.Namespace) -> None:
    _, X, y, feats = load_dataset(cfg, args.test_path or "test")
    model_dir = resolve_arg(cfg, args.model_dir, as_str=True)
    defense_model_dir = resolve_arg(cfg, args.defense_model_dir, as_str=True)
    adv_dir = resolve_arg(cfg, args.adv_dir, as_str=True)
    ens_cfg_dir = resolve_arg(cfg, args.ensemble_config_dir, as_str=True)

    modes = [False, True] if args.at == "both" else [args.at == "true"]

    print("\n" + "=" * 100)
    for s in args.strategy:
        for at in modes:
            res = ensemble_eval(
                cfg, s, at, args.device,
                model_dir, defense_model_dir, adv_dir,
                X, y, feats, config_dir=ens_cfg_dir,
            )
            attacks = [k for k in res if k != "clean"]
            asr_str = "  ".join(f"{a}={res[a].get('asr', 0):.2f}%" for a in attacks)
            mode = "AT" if at else "base"
            clean_f1 = res.get("clean", {}).get("macro_f1", 0)
            print(f"  {s.upper():>7} {mode:>5}  Clean F1={clean_f1:>6.2f}%  {asr_str}")
    print("=" * 100)


def scenarios(cfg: Config, args: argparse.Namespace) -> None:
    targets = list(cfg.all_targets) if "all" in args.target else args.target
    _, X_te, y_te, feats = load_dataset(cfg, args.test_path or "test")

    model_dir = resolve_arg(cfg, args.model_dir, as_str=True)
    defense_model_dir = resolve_arg(cfg, args.defense_model_dir, as_str=True)
    adv_dir = resolve_arg(cfg, args.adv_dir, as_str=True)
    ens_cfg_dir = resolve_arg(cfg, args.ensemble_config_dir, as_str=True)

    preproc = None
    if args.preproc_defense == "feature_squeezing":
        preproc = parse_fs_config(args.fs_config, list(cfg.all_targets),
                                  default_bit=args.bit_depth,
                                  clip_values=(0.0, 1.0))

    rows = run_scenarios(
        cfg, targets=targets, strategies=args.strategy,
        scenarios=args.scenarios,
        X_te=X_te, y_te=y_te, feats=feats,
        device=args.device,
        model_dir=model_dir, defense_model_dir=defense_model_dir,
        adv_dir=adv_dir,
        ensemble_config_dir=ens_cfg_dir, defense=args.defense,
        preprocessing_defence=preproc,
    )

    if not args.no_print_summary:
        print_scenario_summary(rows)

    formats = parse_formats(args.export_format)
    if args.export:
        write_results(rows, resolve_arg(cfg, args.export),
                      label=args.export_label, append=False,
                      scen_meta=SCEN_META, exp_id_fn=exp_id, formats=formats)
    if args.export_into:
        if not args.export_label:
            raise SystemExit("--export-into requires --export-label")
        merge_formats = tuple(formats)
        if "json" not in merge_formats:
            merge_formats = ("json",) + merge_formats
        write_results(rows, resolve_arg(cfg, args.export_into),
                      label=args.export_label, append=True,
                      scen_meta=SCEN_META, exp_id_fn=exp_id, formats=merge_formats)


def register(sub: argparse._SubParsersAction, cfg: Config) -> None:
    targets = list(cfg.all_targets)
    choices = targets + list(cfg.surrogate_targets) + ["all"]

    # eval
    p = sub.add_parser("eval", help="single-model evaluation (clean + adversarial ASR)")
    p.add_argument("--target", "-t", nargs="+", required=True, choices=choices)
    p.add_argument("--at", default="false", choices=["false", "true", "both"])
    add_io_args(p, test=True, model=True, defense_model=True, adv=True)
    add_runtime_args(p)
    p.set_defaults(func=single)

    # eval-ensemble
    p = sub.add_parser("eval-ensemble", help="ensemble evaluation (static / mi4 / mi5)")
    p.add_argument("--strategy", nargs="+", default=["static", "mi4"],
                   choices=["static", "mi4", "mi5"])
    p.add_argument("--at", default="false", choices=["false", "true", "both"])
    add_io_args(p, test=True, model=True, defense_model=True, adv=True, ensemble_cfg=True)
    add_runtime_args(p)
    p.set_defaults(func=ensemble)

    # eval-scenarios
    p = sub.add_parser("eval-scenarios",
                       help="4-scenario eval (S1/S2 single + E1/E2 ensemble)")
    p.add_argument("--target", "-t", nargs="+", default=targets, choices=choices)
    p.add_argument("--strategy", nargs="+", default=["static", "mi4"],
                   choices=["static", "mi4", "mi5"])
    p.add_argument("--scenarios", nargs="+",
                   default=["S1", "S2", "E1", "E2"],
                   choices=["S1", "S2", "E1", "E2"])
    p.add_argument("--defense", default="at", choices=["at", "pgd_at", "distill"],
                   help="defense suffix to load for AT scenarios (default: at)")
    p.add_argument("--preproc-defense", default=None,
                   choices=["feature_squeezing"],
                   help="optional input-preprocessing defense applied before model")
    p.add_argument("--bit-depth", type=int, default=4,
                   help="bit depth for feature_squeezing (default: 4)")
    p.add_argument("--fs-config", default=None,
                   help="per-model FS bit_depth, eg. 'rf=2,et=4' (default: --bit-depth)")
    p.add_argument("--export", default=None,
                   help="write per-format files <base>.<fmt> (overwrite)")
    p.add_argument("--export-into", default=None,
                   help="append into shared <base>.<fmt>; needs --export-label")
    p.add_argument("--export-label", default=None,
                   help="defense label (e.g. 'B0_baseline'); required with --export-into")
    p.add_argument("--export-format", default="md,txt",
                   help="comma-separated formats: md,txt,json (default: md,txt)")
    p.add_argument("--no-print-summary", action="store_true")
    add_io_args(p, test=True, model=True, defense_model=True, adv=True, ensemble_cfg=True)
    add_runtime_args(p)
    p.set_defaults(func=scenarios)
