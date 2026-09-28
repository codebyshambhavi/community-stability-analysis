"""Config loading. All experiment parameters live in config.yaml."""
from __future__ import annotations

from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent


def load_config(path: str | Path | None = None) -> dict:
    path = Path(path) if path else REPO_ROOT / "config.yaml"
    with open(path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    cfg["_repo_root"] = str(REPO_ROOT)
    return cfg


def run_seeds(cfg: dict) -> list[int]:
    """Seeds 0..N-1 (same integer sequence reused for every algorithm and graph)."""
    start = int(cfg["runs"]["seed_start"])
    return list(range(start, start + int(cfg["runs"]["n_runs"])))


def resolve(cfg: dict, key: str) -> Path:
    """Resolve a path from cfg['paths'] relative to the repo root."""
    return REPO_ROOT / cfg["paths"][key]
