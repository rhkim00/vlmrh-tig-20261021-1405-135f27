"""YAML config loading with dotted-key overrides (``--set a.b=c``)."""

from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _parse_value(raw: str) -> Any:
    # YAML parsing gives ints/floats/bools/lists for free ("1e-5" -> float, "[42,43]" -> list).
    return yaml.safe_load(raw)


def apply_overrides(cfg: dict, overrides: list[str] | None) -> dict:
    cfg = copy.deepcopy(cfg)
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"Override must look like key.sub=value, got {item!r}")
        key, raw = item.split("=", 1)
        node = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            if p not in node or not isinstance(node[p], dict):
                raise KeyError(f"Unknown config section {p!r} in override {item!r}")
            node = node[p]
        if parts[-1] not in node:
            raise KeyError(f"Unknown config key {key!r} (refusing to create new keys silently)")
        node[parts[-1]] = _parse_value(raw)
    return cfg


def load_config(path: str | os.PathLike, overrides: list[str] | None = None) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    cfg = apply_overrides(cfg, overrides)
    cfg["_config_path"] = str(path)
    return cfg


def resolve(cfg: dict, rel: str) -> Path:
    """Resolve a project-relative path from the config."""
    p = Path(rel)
    return p if p.is_absolute() else PROJECT_ROOT / p


def output_dir(cfg: dict) -> Path:
    return resolve(cfg, cfg["project"]["output_dir"])


def setup_hf_env(cfg: dict) -> None:
    """Point the HF cache to the project storage (home disk is small). Must run before HF imports load."""
    os.environ.setdefault("HF_HOME", str(resolve(cfg, cfg["project"]["hf_home"])))


def config_hash(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:12]


def dump_config(cfg: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump({k: v for k, v in cfg.items() if not k.startswith("_")}, f, sort_keys=False, allow_unicode=True)
