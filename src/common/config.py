from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    path = Path(config_path) if config_path else project_root() / "configs" / "default.json"
    if not path.is_absolute():
        path = project_root() / path
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    extends = data.pop("extends", None)
    if extends:
        parent_path = Path(extends)
        if not parent_path.is_absolute():
            parent_path = path.parent / parent_path
            if not parent_path.exists():
                parent_path = project_root() / extends
        parent = load_config(parent_path)
        return deep_update(parent, data)
    return data


def resolve_path(config: dict[str, Any], *parts: str) -> Path:
    root = project_root()
    path = root.joinpath(*parts)
    return path


def configured_path(config: dict[str, Any], key: str) -> Path:
    return resolve_path(config, config["paths"][key])


def ensure_project_dirs(config: dict[str, Any]) -> None:
    for key in ("processed_dir", "figures_dir", "metrics_dir", "models_dir", "reports_dir"):
        configured_path(config, key).mkdir(parents=True, exist_ok=True)


def deep_update(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_update(merged[key], value)
        else:
            merged[key] = value
    return merged


def raw_file_path(config: dict[str, Any], dataset_key: str) -> Path:
    raw_dir = configured_path(config, "raw_data_dir")
    return raw_dir / config["dataset"][dataset_key]
