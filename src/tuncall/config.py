"""Load YAML run configs into Pydantic models (D-87).

A config file may name a parent with ``base: <relative path>``. The parent is
loaded first and the child is deep-merged over it: nested mappings merge key by
key, every other value (lists included) is replaced.

Define config sections as subclasses of ``StrictModel`` so unknown or misspelled
keys are errors. Log the resolved config with ``cfg.model_dump(mode="json")``.
"""

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict


class StrictModel(BaseModel):
    """Base for config models: unknown keys raise a validation error."""

    model_config = ConfigDict(extra="forbid")


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Return ``base`` with ``override`` merged in; neither input is modified."""
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_yaml(path: str | Path, _seen: frozenset[Path] = frozenset()) -> dict[str, Any]:
    """Read a YAML config, resolving its ``base:`` chain."""
    path = Path(path).resolve()
    if path in _seen:
        raise ValueError(f"config inheritance cycle at {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path}: top level must be a mapping")
    base = data.pop("base", None)
    if base is None:
        return data
    return deep_merge(load_yaml(path.parent / base, _seen | {path}), data)


def load_config[M: BaseModel](path: str | Path, model: type[M]) -> M:
    """Load a YAML config (with its ``base:`` chain) and validate it as ``model``."""
    return model.model_validate(load_yaml(path))
