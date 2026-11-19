"""Small typed YAML configuration loader with dotted command-line overrides."""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, TypeVar, get_type_hints

import yaml


@dataclass
class DataConfig:
    manifest: str = ""
    manifests: list[str] = field(default_factory=list)
    val_manifests: list[str] = field(default_factory=list)
    processed_root: str = "data/processed"
    image_size: int = 512
    d_max: int = 32
    target_inplane_spacing_mm: float | None = None
    target_depth_spacing_mm: float | None = None
    max_native_depth_spacing_mm: float = 5.0
    feature_root: str | None = None


@dataclass
class RuntimeConfig:
    seed: int = 42
    max_len: int = 4096
    num_workers: int = 6
    pin_memory: bool = True
    persistent_workers: bool = True
    prefetch_factor: int = 2


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    model: dict[str, Any] = field(default_factory=dict)
    train: dict[str, Any] = field(default_factory=dict)
    eval: dict[str, Any] = field(default_factory=dict)
    sources: dict[str, Any] = field(default_factory=dict)


T = TypeVar("T")


def _coerce(cls: type[T], values: Mapping[str, Any]) -> T:
    kwargs: dict[str, Any] = {}
    hints = get_type_hints(cls)
    for f in dataclasses.fields(cls):
        if f.name not in values:
            continue
        value = values[f.name]
        field_type = hints.get(f.name, f.type)
        if dataclasses.is_dataclass(field_type) and isinstance(value, Mapping):
            value = _coerce(field_type, value)
        kwargs[f.name] = value
    return cls(**kwargs)


def _set_dotted(root: dict[str, Any], dotted: str, value: Any) -> None:
    keys = dotted.split(".")
    node = root
    for key in keys[:-1]:
        child = node.setdefault(key, {})
        if not isinstance(child, dict):
            raise ValueError(f"override crosses non-mapping key: {dotted}")
        node = child
    node[keys[-1]] = value


def load_config(path: str | Path, overrides: list[str] | None = None) -> Config:
    raw = yaml.safe_load(Path(path).read_text()) or {}
    if not isinstance(raw, dict):
        raise ValueError("configuration root must be a mapping")
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"override must be a.b=value: {item}")
        key, text = item.split("=", 1)
        _set_dotted(raw, key, yaml.safe_load(text))
    return _coerce(Config, raw)
