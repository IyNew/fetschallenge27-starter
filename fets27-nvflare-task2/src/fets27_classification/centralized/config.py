"""YAML configuration loading and dotted-key overrides."""

from __future__ import annotations

import os
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


def load_config(
    path: str | Path, overrides: list[str] | None = None
) -> dict[str, Any]:
    """Load YAML, apply ``FETS27_DATA_ROOT``, then apply dotted overrides.

    The environment override makes the configuration set portable to an HPC
    filesystem without rewriting every model and sweep YAML file. Explicit
    dotted overrides retain highest precedence for one-off commands.

    Args:
        path: Path to the YAML configuration file.
        overrides: Optional ``key=value`` dotted overrides.

    Returns:
        The resolved configuration mapping.

    Raises:
        ValueError: If an override is not in ``key=value`` form.
    """
    with Path(path).open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    data_root = os.environ.get("FETS27_DATA_ROOT")
    if data_root:
        config.setdefault("data", {})["data_root"] = data_root
    for item in overrides or []:
        key, separator, value = item.partition("=")
        if not separator or not key:
            raise ValueError(f"Override must be key=value, got {item!r}.")
        target = config
        parts = key.split(".")
        for part in parts[:-1]:
            target = target.setdefault(part, {})
            if not isinstance(target, dict):
                raise ValueError(f"Cannot override nested key {key!r}.")
        target[parts[-1]] = yaml.safe_load(value)
    return config


def with_overrides(config: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """Return a deep-copied configuration with dotted mapping overrides applied.

    Args:
        config: The base configuration mapping.
        overrides: Mapping of dotted keys to replacement values.

    Returns:
        A new configuration mapping with the overrides applied.
    """
    result = deepcopy(config)
    for key, value in overrides.items():
        target = result
        parts = key.split(".")
        for part in parts[:-1]:
            target = target.setdefault(part, {})
        target[parts[-1]] = value
    return result
