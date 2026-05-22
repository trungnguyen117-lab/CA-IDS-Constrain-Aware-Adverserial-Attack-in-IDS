"""Result export — 3-layer pipeline:

1. **transform** (data → data): build_overview / build_defense_block / merge_data
2. **render**    (data → str):  render_json / render_markdown / render_text
3. **I/O**       (str → file):  write_results

Each layer is independently testable. Renderers never read disk; transform
never touches strings; I/O never compute statistics.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

# ── Formatting helper (render-layer primitive) ───────────────────────────────


def fmt_value(v) -> str:
    if v is None:
        return "—"
    if isinstance(v, bool):
        return "✓" if v else "✗"
    if isinstance(v, float):
        return f"{v:.2f}"
    return str(v)


# ── 1. Transform layer ───────────────────────────────────────────────────────


def build_overview(rows, label, scen_meta, exp_id_fn) -> list[dict]:
    """Flat list — one row per (scenario, target/strategy).

    Identification + clean stats only (no aggregate means; per-attack details
    live in the defense block).
    """
    overview = []
    for scen, key, res in rows:
        meta = scen_meta[scen]
        clean = res.get("clean", {})
        overview.append({
            "defense": label,
            "exp_id": exp_id_fn(scen, key, meta["scope"], label=label),
            "scenario": scen,
            "scenario_desc": meta["desc"],
            "scope": meta["scope"],
            "at": meta["at"],
            "target": key,
            "clean_acc": clean.get("acc"),
            "clean_f1": clean.get("macro_f1"),
            "clean_prec": clean.get("macro_prec"),
            "clean_dr": clean.get("macro_dr"),
        })
    return overview


def build_defense_block(rows, scen_meta) -> dict:
    """Nested dict — {scenarios, scope, targets, attacks, clean, results}."""
    targets: list = []
    attacks_seen: set = set()
    clean_by_target: dict = {}
    results: dict = {}
    scenarios: list = []

    for scen, key, res in rows:
        if scen not in scenarios:
            scenarios.append(scen)
        if key not in targets:
            targets.append(key)
        clean = res.get("clean", {})
        clean_by_target[key] = clean
        for atk, vals in res.items():
            if atk == "clean":
                continue
            attacks_seen.add(atk)
            results.setdefault(atk, {})[key] = vals

    return {
        "scenarios": scenarios,
        "scope": scen_meta[scenarios[0]]["scope"] if scenarios else "single",
        "targets": targets,
        "attacks": sorted(attacks_seen),
        "clean": clean_by_target,
        "results": results,
    }


def merge_data(existing: dict | None, label: str,
               overview: list[dict], block: dict) -> dict:
    """Merge a new (overview, block) into existing aggregate data.

    Drops any prior records for ``label`` (re-run replaces wholesale).
    Returns a fresh dict — does not mutate ``existing``.
    """
    data = {
        "overview": list((existing or {}).get("overview", [])),
        "defenses": dict((existing or {}).get("defenses", {})),
    }
    if label:
        data["overview"] = [r for r in data["overview"]
                            if r.get("defense") != label]
        data["defenses"][label] = block
    data["overview"].extend(overview)
    return data


# ── 2. Render layer (data → string, no I/O) ──────────────────────────────────


def render_json(data: dict) -> str:
    return json.dumps(data, indent=2, default=str)


def render_markdown(data: dict) -> str:
    out = ["# Defense comparison results", ""]

    overview = data.get("overview", [])
    if overview:
        out += ["## Overview", ""]
        cols = ["defense", "exp_id", "scenario", "target", "clean_f1"]
        align = {"clean_f1": ":--:"}
        out.append("| " + " | ".join(cols) + " |")
        out.append("|" + "|".join(align.get(c, "---") for c in cols) + "|")
        for r in overview:
            out.append("| " + " | ".join(fmt_value(r.get(c)) for c in cols) + " |")
        out.append("")

    defenses = data.get("defenses", {})
    if defenses:
        out += ["## Per-defense attack breakdown", ""]
        for label, block in defenses.items():
            scen_str = ", ".join(block.get("scenarios", []))
            scope = block.get("scope", "")
            out.append(f"### {label}  *(scenario: {scen_str}, scope: {scope})*")
            out.append("")
            out.append(attack_table_html(block))
            out.append("")

    return "\n".join(out) + "\n"


def render_text(data: dict) -> str:
    from tabulate import tabulate

    out = ["Defense comparison results", "=" * 28, ""]

    overview = data.get("overview", [])
    if overview:
        out += ["## Overview", ""]
        cols = ["defense", "exp_id", "scenario", "target", "clean_f1"]
        rows = [[fmt_value(r.get(c)) for c in cols] for r in overview]
        out.append(tabulate(rows, headers=cols, tablefmt="grid"))
        out.append("")

    defenses = data.get("defenses", {})
    if defenses:
        out += ["", "## Per-defense attack breakdown", ""]
        for label, block in defenses.items():
            scen_str = ", ".join(block.get("scenarios", []))
            scope = block.get("scope", "")
            out.append(f"### {label}  ({scen_str}, {scope})")
            out.append("")
            out.append(attack_table_text(block))
            out.append("")

    return "\n".join(out) + "\n"


_METRIC_COLS = [
    ("acc", "acc"),
    ("macro_f1", "f1"),
    ("macro_prec", "prec"),
    ("macro_dr", "dr"),
    ("asr", "asr"),
]


def attack_table_text(block: dict) -> str:
    """Plaintext attack table — per target: acc | f1 | prec | dr | asr,
    plus ``<target>.eff`` when any attack has effective_asr."""
    from tabulate import tabulate

    targets = block.get("targets", [])
    attacks = block.get("attacks", [])
    clean = block.get("clean", {})
    results = block.get("results", {})

    has_eff = any(
        results.get(atk, {}).get(t, {}).get("effective_asr") is not None
        for atk in attacks for t in targets
    )

    headers = ["attack"]
    for t in targets:
        headers += [f"{t}.{short}" for _, short in _METRIC_COLS]
        if has_eff:
            headers.append(f"{t}.eff")

    rows = []
    row = ["Original"]
    for t in targets:
        c = clean.get(t, {})
        for key, _ in _METRIC_COLS:
            row.append("—" if key == "asr" else fmt_value(c.get(key)))
        if has_eff:
            row.append("—")
    rows.append(row)
    for atk in attacks:
        row = [atk]
        for t in targets:
            r = results.get(atk, {}).get(t, {})
            for key, _ in _METRIC_COLS:
                row.append(fmt_value(r.get(key)))
            if has_eff:
                row.append(fmt_value(r.get("effective_asr")))
        rows.append(row)

    return tabulate(rows, headers=headers, tablefmt="grid")


def attack_table_html(block: dict) -> str:
    """HTML table with two-row header (target merged across acc/asr[/eff])."""
    targets = block.get("targets", [])
    attacks = block.get("attacks", [])
    clean = block.get("clean", {})
    results = block.get("results", {})

    has_eff = any(
        results.get(atk, {}).get(t, {}).get("effective_asr") is not None
        for atk in attacks for t in targets
    )
    cols_per_t = len(_METRIC_COLS) + (1 if has_eff else 0)
    sub_hdrs = "".join(f'<th align="right">{short}</th>'
                       for _, short in _METRIC_COLS)
    if has_eff:
        sub_hdrs += '<th align="right">eff</th>'

    lines = ['<table>', '  <thead>']
    head1 = '    <tr><th rowspan="2">attack</th>'
    for t in targets:
        head1 += f'<th colspan="{cols_per_t}" align="center">{t}</th>'
    head1 += '</tr>'
    lines.append(head1)
    head2 = '    <tr>' + (sub_hdrs * len(targets)) + '</tr>'
    lines.append(head2)
    lines += ['  </thead>', '  <tbody>']

    row = '    <tr><td><i>Original</i></td>'
    for t in targets:
        c = clean.get(t, {})
        for key, _ in _METRIC_COLS:
            v = "—" if key == "asr" else fmt_value(c.get(key))
            row += f'<td align="right">{v}</td>'
        if has_eff:
            row += '<td align="right">—</td>'
    row += '</tr>'
    lines.append(row)

    for atk in attacks:
        row = f'    <tr><td>{atk}</td>'
        for t in targets:
            r = results.get(atk, {}).get(t, {})
            for key, _ in _METRIC_COLS:
                row += f'<td align="right">{fmt_value(r.get(key))}</td>'
            if has_eff:
                row += f'<td align="right">{fmt_value(r.get("effective_asr"))}</td>'
        row += '</tr>'
        lines.append(row)

    lines += ['  </tbody>', '</table>']
    return "\n".join(lines)


# ── 3. I/O layer (str → file) ────────────────────────────────────────────────


_RENDERERS = {
    "json": (".json", render_json),
    "md":   (".md",   render_markdown),
    "txt":  (".txt",  render_text),
}


def write_results(rows, path, label, *, append: bool, scen_meta, exp_id_fn,
                  formats: Iterable[str] = ("json", "md", "txt")) -> dict[str, Path]:
    """Build → render → write. ``formats`` selects which files to emit.

    Append mode: read the existing JSON beside ``path`` (if any) to merge with;
    if JSON not in ``formats`` the merge still happens via the on-disk JSON.

    Returns ``{format: path}`` for files actually written.
    """
    base = Path(path)
    if base.suffix in {".xlsx", ".json", ".md", ".txt"}:
        base = base.with_suffix("")
    base.parent.mkdir(parents=True, exist_ok=True)

    # Load existing JSON for append (independent of `formats` choice)
    existing = None
    if append:
        json_path = base.with_suffix(".json")
        if json_path.is_file():
            existing = json.loads(json_path.read_text())

    # Transform
    overview = build_overview(rows, label, scen_meta, exp_id_fn)
    block = build_defense_block(rows, scen_meta)
    data = merge_data(existing, label, overview, block)

    # Render + write only the requested formats
    written: dict[str, Path] = {}
    for fmt in formats:
        if fmt not in _RENDERERS:
            raise ValueError(f"Unknown format {fmt!r}; pick from {list(_RENDERERS)}")
        ext, renderer = _RENDERERS[fmt]
        out_path = base.with_suffix(ext)
        out_path.write_text(renderer(data))
        written[fmt] = out_path
    return written


