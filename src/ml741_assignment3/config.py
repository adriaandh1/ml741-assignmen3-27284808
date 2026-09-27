"""Configuration loading and validation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML mapping from disk."""
    config_path = Path(path)
    with config_path.open(encoding="utf-8") as stream:
        loaded = yaml.safe_load(stream)
    if not isinstance(loaded, dict):
        raise ValueError(f"configuration must contain a mapping: {config_path}")
    return loaded


def require_keys(config: dict[str, Any], *keys: str) -> None:
    """Raise a useful error when required top-level keys are absent."""
    missing = [key for key in keys if key not in config]
    if missing:
        raise ValueError(f"missing configuration keys: {', '.join(missing)}")
