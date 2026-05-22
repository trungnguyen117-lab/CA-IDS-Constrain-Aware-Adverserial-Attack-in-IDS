"""Pretty printers for CLI summaries."""

from __future__ import annotations


def print_summary(results: dict) -> None:
    if not results:
        print("No results.")
        return
    sample = next(iter(results.values()))
    attacks = [k for k in sample if k != "clean"]
    print("\n" + "=" * 100)
    print(f"{'Target':<8} {'Clean Acc':>10} {'Clean F1':>10}  " + " ".join(f"{a:>10}" for a in attacks))
    print("-" * 100)
    for t, res in results.items():
        c = res.get("clean", {})
        line = f"{t.upper():<8} {c.get('acc', 0):>9.2f}%  {c.get('macro_f1', 0):>9.2f}%  "
        for a in attacks:
            asr = res.get(a, {}).get("asr")
            line += f"   {asr:>6.2f}% " if asr is not None else "       N/A "
        print(line)
    print("=" * 100)


def print_compare(base: dict, at: dict) -> None:
    print("\n" + "=" * 100)
    print(f"{'Target':<8} {'Mode':<6}  {'Clean F1':>10}  ASR per attack")
    print("-" * 100)
    for t in base:
        for mode, src in (("base", base), ("AT", at)):
            res = src.get(t, {})
            c = res.get("clean", {})
            atks = {k: v for k, v in res.items() if k != "clean"}
            asr = "  ".join(f"{a}={v.get('asr', 0):.2f}%" for a, v in atks.items())
            print(f"{t.upper():<8} {mode:<6}  {c.get('macro_f1', 0):>9.2f}%   {asr}")
    print("=" * 100)


def print_scenario_summary(rows: list) -> None:
    print("\n" + "=" * 100)
    print(f"  {'Scenario':<14} {'Tgt/Strat':<10} {'Clean F1':>9}  ASR per attack")
    print("-" * 100)
    for scen, t, res in rows:
        cf1 = res.get("clean", {}).get("macro_f1", 0)
        atks = {k: v for k, v in res.items() if k != "clean"}
        asr = "  ".join(f"{a}={v.get('asr', 0):.1f}%" for a, v in atks.items())
        print(f"  {scen:<14} {t:<10} {cf1:>8.2f}%  {asr}")
    print("=" * 100)
