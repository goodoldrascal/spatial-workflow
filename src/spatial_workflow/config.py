"""Configuration loading and path resolution."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


_REQUIRED_SECTIONS = ("project", "paths", "schema")


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a workflow YAML file and validate its shared sections."""

    config_path = Path(path).expanduser()
    with config_path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    if not isinstance(config, dict):
        raise ValueError("Configuration must be a YAML mapping")

    for section in _REQUIRED_SECTIONS:
        if not isinstance(config.get(section), dict):
            raise ValueError(f"Configuration section '{section}' must be a mapping")

    if "runtime" in config and not isinstance(config["runtime"], dict):
        raise ValueError("Configuration section 'runtime' must be a mapping")

    return config


def resolve_path(
    config_path: str | Path,
    value: str | Path,
    root: str | Path | None = None,
) -> Path:
    """Resolve a path relative to the config file or a config-relative root."""

    config_dir = Path(config_path).expanduser().resolve().parent
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()

    if root is None:
        base = config_dir
    else:
        base = Path(root).expanduser()
        if not base.is_absolute():
            base = config_dir / base

    return (base / path).resolve()
