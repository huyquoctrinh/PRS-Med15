"""Offline cache of frozen slice-encoder features.

The MedSAM encoder is frozen and applied independently to every slice, so the features of any
training slab are rows ``[start:stop]`` of the full-volume feature array. Each scan is stored
once as ``<feature_root>/<source>/<key>.npy`` with shape ``[D_full, h, w, C]`` (float16) and read
with mmap, so a 32-slice slab only touches its own rows.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import torch

FEATURE_VERSION = "medsam_vit_b_512_v1"


def feature_path(root: str | Path, source: str, image_path: str | Path,
                 geometry: dict[str, Any] | None = None) -> Path:
    """Stable feature file for one scan.

    Volumes reuse the processed-cache key (it already hashes the preprocessing config); native
    2D images are keyed by their path.
    """
    cache_path = (geometry or {}).get("cache_path") if isinstance(geometry, dict) else None
    if cache_path:
        key = Path(cache_path).stem
    else:
        key = hashlib.sha256(f"{FEATURE_VERSION}|{image_path}".encode()).hexdigest()[:40]
    return Path(root) / source / f"{key}.npy"


def load_feature_rows(path: Path, start: int, stop: int, expected_depth: int) -> torch.Tensor | None:
    """Return float16 features ``[stop-start, h, w, C]`` or ``None`` if unusable."""
    try:
        array = np.load(path, mmap_mode="r")
    except (FileNotFoundError, ValueError, OSError):
        return None
    if array.ndim != 4 or array.shape[0] != expected_depth:
        return None
    return torch.from_numpy(np.array(array[start:stop], copy=True))


def save_features(path: Path, value: np.ndarray) -> None:
    """Atomic write so an interrupted extraction never leaves a truncated file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.npy")
    np.save(tmp, value.astype(np.float16, copy=False))
    tmp.replace(path)
