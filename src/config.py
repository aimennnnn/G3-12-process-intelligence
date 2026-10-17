"""Shared helpers: config loading and .env loading (no extra dependencies)."""
from __future__ import annotations

import os
from pathlib import Path

import yaml


def load_config(path: str | Path = "config.yaml") -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_dotenv(path: str | Path = ".env") -> None:
    """Read KEY=value lines from .env into os.environ (existing vars win)."""
    p = Path(path)
    if not p.exists():
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))
