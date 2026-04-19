"""YAML config loading and merging."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config file."""

    with open(path) as f:
        return yaml.safe_load(f) or {}


def merge_configs(*configs: dict) -> dict[str, Any]:
    """
    Merge multiple config dicts, later ones override earlier.

    Performs shallow merge at top level, deep merge for nested dicts.
    """
    result: dict[str, Any] = {}
    for config in configs:
        for key, value in config.items():
            if (
                key in result
                and isinstance(result[key], dict)
                and isinstance(value, dict)
            ):
                result[key] = merge_configs(result[key], value)
            else:
                result[key] = value
    return result


def load_and_merge(*paths: str | Path) -> dict[str, Any]:
    """Load and merge multiple YAML config files."""
    configs = [load_config(p) for p in paths]
    return merge_configs(*configs)
