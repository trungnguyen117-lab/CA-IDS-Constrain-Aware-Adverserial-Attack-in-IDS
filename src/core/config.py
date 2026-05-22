"""Pydantic-typed dataset config — single source of truth.

Each dataset has ``<dataset>/config.yaml``. ``Config.from_yaml`` loads it
and exposes typed fields directly. YAML is allowed to keep the nested
``models:`` / ``attacks:`` / ``tune:`` blocks for readability — a
validator lifts those keys to top level on load.

Usage::

    from src.utils.config import Config
    cfg = Config.from_yaml("IIOT-2021/config.yaml")
    cfg.paths.train         # Path (absolute, leak-mode aware ablation lives in
                             #       a separate config file, not env vars)
    cfg.tree_targets        # list[str]
    cfg.wb_attacks          # list[str]
    cfg.de_iter             # int (was cfg.tune.de_iter)
"""

from __future__ import annotations

import sys
from importlib import import_module
from pathlib import Path
from typing import ClassVar

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


def _deep_merge(base: dict, over: dict) -> dict:
    """Recursively merge ``over`` into ``base``. Lists/scalars in ``over`` replace."""
    out = dict(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _load_with_extends(path: Path) -> dict:
    """Load YAML with optional ``extends: <relative-yaml>`` chain support."""
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    parent_ref = data.pop("extends", None)
    if not parent_ref:
        return data
    parent_path = (path.parent / parent_ref).resolve()
    parent_data = _load_with_extends(parent_path)
    return _deep_merge(parent_data, data)


# ── Paths sub-model (kept: 17+ fields + extras + get()) ─────────────────────


class Paths(BaseModel):
    """File and directory paths. Required: ``train``, ``test``."""

    model_config = ConfigDict(extra="allow")  # tolerate dataset-specific extras

    train: Path
    test: Path
    models: Path = Path("baseline/models")
    models_at: Path = Path("defense/at")
    models_pgd_at: Path = Path("defense/pgd_at")
    models_distill: Path = Path("defense/distill")
    magnet: Path = Path("defense/magnet")
    adv_eval: Path = Path("adv_samples/adv_eval")
    adv_training: Path = Path("adv_samples/adv_training")
    report: Path = Path("report")

    # IIOT-style preprocessing pipeline outputs (optional)
    raw: Path | None = None
    clean: Path | None = None
    clean_noleak: Path | None = None
    train_real: Path | None = None
    train_for_shap: Path | None = None
    train_shap_real: Path | None = None
    test_real: Path | None = None

    def get(self, key: str) -> Path | None:
        """Generic dynamic-key access (extras / arbitrary keys).
        Prefer direct attribute access (``cfg.paths.train``)."""
        v = getattr(self, key, None)
        if v is not None:
            return v
        extras = getattr(self, "__pydantic_extra__", None) or {}
        return extras.get(key)


# Backward-compat alias
PathsConfig = Paths


# ── Top-level config ────────────────────────────────────────────────────────


class Config(BaseModel):
    """All dataset config in one flat class.

    YAML may keep nested ``models:`` / ``attacks:`` / ``tune:`` blocks for
    readability — the validator below lifts those keys to top level so
    callers access them flat (``cfg.tree_targets``, ``cfg.wb_attacks``,
    ``cfg.de_iter``).
    """

    model_config = ConfigDict(extra="ignore", arbitrary_types_allowed=True)

    # Identity
    name: str = ""           # falls back to yaml parent dir if empty
    label_col: str = "Label"
    n_classes: int = 0
    label_names: list[str] = []

    # Continuous (float) features. Empty → fallback to binary-only protected
    # detection. When set, InputNorm normalizes only these and gen-adv protects
    # all non-cont features.
    cont_features: list[str] = []

    # Paths (sub-model — 17+ fields with extras-allowed legacy aliases)
    paths: Paths

    # Models (was ModelsConfig)
    tree_targets: list[str] = []
    dl_targets: list[str] = []
    surrogate_targets: list[str] = []
    ext: dict[str, str] = {}
    # Map DL/surrogate target → "module.ClassName". Lazy via :meth:`dl_factory`.
    dl_factory_map: dict[str, str] = {}

    # Attacks (was AttacksConfig)
    wb_attacks: list[str] = []
    bb_attacks: list[str] = []
    attack_compat: dict[str, list[str]] = {}
    transfer_sources: dict[str, list[str]] = {}

    # Adversarial training
    at_weights: dict[str, dict[str, float]] = {}

    # Ensemble tuning hyperparams (was TuneConfig). CLI flags fall back here.
    de_iter: int = 200
    de_popsize: int = 25
    tune_seed: int = 42

    # Preprocessing (IIOT only — optional dicts)
    drop_cols: list[str] = []
    label: dict = Field(default_factory=dict)
    augment: dict = Field(default_factory=dict)
    covas: dict = Field(default_factory=dict)
    shap: dict = Field(default_factory=dict)
    split: dict = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def coerce_optional_lists(cls, data):
        """YAML ``leak_cols:`` with no value parses as None — coerce to []."""
        if isinstance(data, dict):
            for k in ("drop_cols", "label_names"):
                if data.get(k) is None and k in data:
                    data[k] = []
        return data

    # Dataset root dir (parent of config.yaml). Set by ``from_yaml``.
    root: Path = Path(".")

    # ── Validator: lift YAML nested blocks to top-level ────────────────────
    @model_validator(mode="before")
    @classmethod
    def flatten_blocks(cls, data):
        """Lift nested YAML blocks to flat fields.

        Rename map (YAML → flat field):
          models.dl_factory → dl_factory_map
          attacks.wb        → wb_attacks
          attacks.bb        → bb_attacks
          attacks.compat    → attack_compat
          tune.seed         → tune_seed
        Other keys keep their name.
        """
        if not isinstance(data, dict):
            return data
        # models.X → top-level.
        m = data.pop("models", None)
        if isinstance(m, dict):
            rename = {
                "tree": "tree_targets",
                "dl": "dl_targets",
                "surrogate": "surrogate_targets",
                "dl_factory": "dl_factory_map",
            }
            for k, v in m.items():
                data.setdefault(rename.get(k, k), v)
        # attacks.X → top-level (rename: wb→wb_attacks, bb→bb_attacks, compat→attack_compat)
        a = data.pop("attacks", None)
        if isinstance(a, dict):
            rename = {"wb": "wb_attacks", "bb": "bb_attacks", "compat": "attack_compat"}
            for k, v in a.items():
                data.setdefault(rename.get(k, k), v)
        # tune.X → top-level (rename: seed → tune_seed)
        t = data.pop("tune", None)
        if isinstance(t, dict):
            rename = {"seed": "tune_seed"}
            for k, v in t.items():
                data.setdefault(rename.get(k, k), v)
        return data

    # ── Loader ────────────────────────────────────────────────────────────
    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        p = Path(path).resolve()
        # Make per-dataset ``model_local.*`` importable (used by dl_factory).
        if str(p.parent) not in sys.path:
            sys.path.insert(0, str(p.parent))
        data = _load_with_extends(p)
        cfg = cls(**data)
        cfg.root = p.parent
        if not cfg.name:
            cfg.name = cfg.root.name
        # Resolve relative paths against the dataset root.
        for field in Paths.model_fields:
            v = getattr(cfg.paths, field, None)
            if v is not None and not v.is_absolute():
                setattr(cfg.paths, field, (cfg.root / v).resolve())
        extras = getattr(cfg.paths, "__pydantic_extra__", None) or {}
        for key, v in list(extras.items()):
            if isinstance(v, (str, Path)):
                rel = Path(v)
                extras[key] = rel if rel.is_absolute() else (cfg.root / rel).resolve()
        return cfg

    # ── Properties ────────────────────────────────────────────────────────
    @property
    def all_targets(self) -> list[str]:
        return list(self.tree_targets) + list(self.dl_targets)

    @property
    def split_seed(self) -> int:
        """Preprocessing seed (split.random_state, fallback augment.random_state)."""
        return int(self.split.get("random_state",
                                  self.augment.get("random_state", 42)))

    # ── Target predicates / model file helpers ────────────────────────────
    def is_tree(self, target: str) -> bool:
        return target in self.tree_targets

    def is_dl(self, target: str) -> bool:
        return target in self.dl_targets or target in self.surrogate_targets

    def ext_of(self, target: str) -> str:
        if target in self.ext:
            return self.ext[target]
        return ".pkl" if self.is_tree(target) else ".pth"

    def model_stem(self, target: str, defense: str | None = None) -> str:
        """Filename without ext. Baseline: ``{target}``. Defense: ``{target}_{defense}``."""
        return target if defense is None else f"{target}_{defense}"

    # Per-defense default output dir. Caller can override via ``base_dir``.
    DEFENSE_DIRS: ClassVar[dict[str, str]] = {
        "at":      "models_at",
        "pgd_at":  "models_pgd_at",
        "distill": "models_distill",
    }

    def model_path(self, target: str, defense: str | None = None,
                   base_dir: str | Path | None = None) -> Path:
        if base_dir is not None:
            base = Path(base_dir)
        elif defense is None:
            base = self.paths.models
        else:
            attr = self.DEFENSE_DIRS.get(defense)
            if attr is None:
                raise ValueError(f"Unknown defense '{defense}'; known: {list(self.DEFENSE_DIRS)}")
            base = getattr(self.paths, attr)
        return base / f"{self.model_stem(target, defense)}{self.ext_of(target)}"

    # ── Sub-config + DL factory ───────────────────────────────────────────
    def cfg_yaml(self, group: str, name: str) -> dict:
        """Read sub-yaml at ``<root>/config/<group>/<name>.yaml``."""
        path = self.root / "config" / group / f"{name}.yaml"
        if not path.is_file():
            return {}
        with open(path) as f:
            return yaml.safe_load(f) or {}

    def dl_factory(self, name: str):
        cls_path = self.dl_factory_map.get(name)
        if not cls_path:
            return None
        module_name, cls_name = cls_path.rsplit(".", 1)
        return getattr(import_module(module_name), cls_name)()

    def resolve(self, p: str | Path | None) -> Path | None:
        if p is None:
            return None
        p = Path(p)
        return p if p.is_absolute() else (self.root / p).resolve()

    # ── Adversarial sample path helpers ───────────────────────────────────
    # Methods renamed with ``_path`` suffix to avoid colliding with
    # ``self.paths.adv_eval`` / ``self.paths.adv_training`` field names.
    def adv_eval_path(self, target: str = "", filename: str = "") -> Path:
        base = self.paths.adv_eval
        if not target:
            return base
        return base / target / filename if filename else base / target

    def adv_train_path(self, target: str = "", filename: str = "") -> Path:
        base = self.paths.adv_training
        if not target:
            return base
        return base / target / filename if filename else base / target

    def adv_csv(self, target: str, attack: str, source: str = "test",
                base: str | Path | None = None) -> Path:
        suffix = "train_adv" if source == "train" else "adv"
        fname = f"{target}_{attack}_{suffix}.csv"
        if base is not None:
            sub = "adv_training" if source == "train" else "adv_eval"
            return Path(base) / sub / target / fname
        return (self.adv_train_path(target, fname) if source == "train"
                else self.adv_eval_path(target, fname))
