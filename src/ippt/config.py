"""Paths and YAML config loading."""

from __future__ import annotations

import os
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "configs"
SCORING_DIR = PROJECT_ROOT / "data" / "scoring"


def data_dir() -> Path:
    """Root for downloaded datasets and models. Defaults to data/raw/ in the project (gitignored)."""
    return Path(os.environ.get("IPPT_DATA_DIR", PROJECT_ROOT / "data" / "raw"))


def interim_dir() -> Path:
    """Derived caches (pose tracks, web-playable transcodes). Gitignored."""
    return PROJECT_ROOT / "data" / "interim"


def models_dir() -> Path:
    return data_dir() / "models"


def load_yaml(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_pipeline_config(path: str | Path | None = None) -> dict:
    cfg = load_yaml(CONFIG_DIR / "pipeline.yaml")
    if path:
        cfg = deep_merge(cfg, load_yaml(path))
    return cfg


def load_rules(exercise: str, path: str | Path | None = None) -> dict:
    if exercise not in ("pushup", "situp"):
        raise ValueError(f"unknown exercise {exercise!r}; expected 'pushup' or 'situp'")
    return load_yaml(path or CONFIG_DIR / "rules" / f"{exercise}.yaml")
