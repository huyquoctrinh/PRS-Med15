"""Modality normalization, reversible spatial preprocessing, slabs, and cache."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
from PIL import Image
from scipy import ndimage

from .geometry import (TransformRecord, VolumeTransformRecord, apply_transform,
                       make_letterbox_transform)


CACHE_FORMAT_VERSION = "prs-med15-volume-v2"


def normalize_ct(array: np.ndarray) -> np.ndarray:
    return np.clip((array.astype(np.float32) + 1000.0) / 2000.0, 0.0, 1.0)


def normalize_mr(array: np.ndarray) -> np.ndarray:
    x = array.astype(np.float32)
    finite = x[np.isfinite(x)]
    if finite.size == 0:
        return np.zeros_like(x)
    lo, hi = np.percentile(finite, (0.5, 99.5))
    if hi <= lo:
        return np.zeros_like(x)
    return np.clip((x - lo) / (hi - lo), 0.0, 1.0)


def load_rgb(path: str | Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


def preprocess_native_2d(image: np.ndarray, image_size: int = 512) -> tuple[np.ndarray, np.ndarray, TransformRecord]:
    """Letterbox a native RGB image without distorting aspect ratio."""
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError("native image must be [H,W,3]")
    transform = make_letterbox_transform(image.shape[:2], (image_size, image_size))
    chw = np.transpose(image.astype(np.float32), (2, 0, 1))
    processed = apply_transform(chw, transform, order=1)[:, None]
    valid = apply_transform(np.ones((1,) + image.shape[:2], np.uint8), transform, order=0).astype(bool)
    return processed, valid, transform


@dataclass(frozen=True)
class PreprocessConfig:
    image_size: int = 512
    target_inplane_spacing_mm: float | None = None
    target_depth_spacing_mm: float | None = None
    max_native_depth_spacing_mm: float = 5.0
    d_max: int = 32


@dataclass
class ProcessedVolume:
    image: np.ndarray  # [C,D,H,W]
    valid_mask: np.ndarray  # [D,H,W]
    label: np.ndarray | None
    transform: VolumeTransformRecord
    spacing_mm: tuple[float, float, float]


@dataclass(frozen=True)
class Slab:
    start: int
    stop: int
    image: np.ndarray
    valid_mask: np.ndarray
    label: np.ndarray | None
    visibility: dict[str, bool]


def preprocess_volume(image: np.ndarray, spacing_mm: tuple[float, float, float], modality: str,
                      config: PreprocessConfig = PreprocessConfig(), label: np.ndarray | None = None,
                      source_affine: np.ndarray | None = None,
                      canonical_affine: np.ndarray | None = None,
                      replicate_channels: bool = True) -> ProcessedVolume:
    """Preprocess a [D,H,W] volume without cropping its in-plane field of view.

    ``replicate_channels=False`` returns a single-channel [1,D,H,W] image (the cache stores one
    channel and replicates at load), avoiding a 3x float32 copy of large volumes.
    """
    if image.ndim != 3:
        raise ValueError("volume must have shape [D,H,W]")
    if label is not None and label.shape != image.shape:
        raise ValueError("image/label shape mismatch")
    original_shape = tuple(int(x) for x in image.shape)
    norm = normalize_ct(image) if modality == "CT" else normalize_mr(image)
    zsp, ysp, xsp = map(float, spacing_mm)
    target_xy = config.target_inplane_spacing_mm or min(ysp, xsp)
    xy_zoom = (1.0, ysp / target_xy, xsp / target_xy)
    norm = ndimage.zoom(norm, xy_zoom, order=1, mode="nearest", prefilter=False)
    lab = ndimage.zoom(label, xy_zoom, order=0, mode="nearest", prefilter=False) if label is not None else None
    target_z = config.target_depth_spacing_mm or min(zsp, config.max_native_depth_spacing_mm)
    if not np.isclose(target_z, zsp):
        z_zoom = zsp / target_z
        norm = ndimage.zoom(norm, (z_zoom, 1, 1), order=1, mode="nearest", prefilter=False)
        lab = ndimage.zoom(lab, (z_zoom, 1, 1), order=0, mode="nearest", prefilter=False) if lab is not None else None
    resampled_shape = tuple(int(x) for x in norm.shape)
    letterbox = make_letterbox_transform(norm.shape[-2:], (config.image_size, config.image_size))
    norm = apply_transform(norm, letterbox, order=1)
    lab = apply_transform(lab, letterbox, order=0) if lab is not None else None
    valid = apply_transform(np.ones((norm.shape[0],) + letterbox.original_shape, np.uint8), letterbox, order=0).astype(bool)
    transform = VolumeTransformRecord(
        original_shape=original_shape, resampled_shape=resampled_shape,
        output_shape=tuple(int(x) for x in norm.shape),
        zoom=tuple(r / o for r, o in zip(resampled_shape, original_shape, strict=True)),
        crop=tuple((0, n) for n in original_shape), letterbox=letterbox,
        source_affine=(tuple(tuple(float(x) for x in row) for row in source_affine)
                       if source_affine is not None else None),
        canonical_affine=(tuple(tuple(float(x) for x in row) for row in canonical_affine)
                          if canonical_affine is not None else None),
        unknown_geometry=canonical_affine is None,
    )
    norm = norm.astype(np.float32, copy=False)
    image_c = np.repeat(norm[None], 3, axis=0) if replicate_channels else norm[None]
    return ProcessedVolume(image_c, valid, lab, transform, (target_z, target_xy, target_xy))


def split_slabs(volume: ProcessedVolume, object_masks: dict[str, np.ndarray] | None = None,
                d_max: int = 32) -> list[Slab]:
    depth = volume.image.shape[1]
    masks = object_masks or {}
    slabs: list[Slab] = []
    for start in range(0, depth, d_max):
        stop = min(depth, start + d_max)
        slabs.append(Slab(start, stop, volume.image[:, start:stop], volume.valid_mask[start:stop],
                          volume.label[start:stop] if volume.label is not None else None,
                          {key: bool(mask[start:stop].any()) for key, mask in masks.items()}))
    return slabs


def extract_object_mask(label: np.ndarray | None, label_id: int | None, label_ref: str | None,
                        mask_file: str | None) -> np.ndarray:
    """Extract label-map or TotalSegmentator per-file masks lazily.

    Per-file branch adapted from PRS-Med/data_processing/dialogues/export_viewer.py:object_masks.
    """
    if label_id is not None:
        if label is None:
            raise ValueError("label map required for label_id object")
        return label == int(label_id)
    if mask_file and label_ref:
        path = Path(label_ref) / mask_file
        return np.asarray(nib.as_closest_canonical(nib.load(str(path))).dataobj) > 0
    raise ValueError("object has neither label_id nor mask_file")


def cache_key(path: str | Path, config: PreprocessConfig) -> str:
    path = Path(path)
    payload = {"path": str(path.resolve()), "mtime_ns": path.stat().st_mtime_ns,
               "config": asdict(config), "code_version": CACHE_FORMAT_VERSION}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


class ProcessedCache:
    """Versioned, single-channel NPZ cache for one source's processed scans.

    ``object_masks`` is used only for sources such as TotalSegmentator whose
    annotations are separate files.  A merged int16 label map is retained for
    compact access, while packed per-object masks make overlaps lossless.
    """

    overlap_policy = "int16 label map plus packed per-object masks for lossless overlaps"

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def path_for(self, source_path: str | Path, config: PreprocessConfig) -> Path:
        return self.root / f"{cache_key(source_path, config)}.npz"

    def save(
        self,
        source_path: str | Path,
        config: PreprocessConfig,
        value: ProcessedVolume,
        object_masks: dict[str, np.ndarray] | None = None,
        packed_object_masks: dict[str, np.ndarray] | None = None,
    ) -> Path:
        """Save one processed scan.

        ``packed_object_masks`` holds ``np.packbits(mask.reshape(-1))`` rows (same layout as the
        packing of ``object_masks``) so callers can drop each unpacked mask as soon as it is made.
        """
        self.root.mkdir(parents=True, exist_ok=True)
        path = self.path_for(source_path, config)
        tmp = path.with_suffix(".tmp.npz")
        image = value.image[0] if value.image.ndim == 4 else value.image
        packed = dict(packed_object_masks or {})
        for key, mask in (object_masks or {}).items():
            packed[key] = np.packbits(np.asarray(mask, dtype=bool).reshape(-1))
        mask_keys = sorted(packed)
        if mask_keys:
            packed_masks = np.stack([packed[key] for key in mask_keys])
            mask_shape = np.asarray(value.valid_mask.shape, dtype=np.int32)
        else:
            packed_masks = np.empty((0, 0), dtype=np.uint8)
            mask_shape = np.asarray([], dtype=np.int32)
        metadata = {
            "format_version": CACHE_FORMAT_VERSION,
            "overlap_policy": self.overlap_policy,
            "object_keys": mask_keys,
        }
        np.savez_compressed(
            tmp,
            image=image.astype(np.float16),
            valid_mask=value.valid_mask.astype(np.uint8),
            label=(
                value.label.astype(np.int16, copy=False)
                if value.label is not None
                else np.array([], dtype=np.int16)
            ),
            transform=json.dumps(value.transform.to_dict()),
            spacing=np.asarray(value.spacing_mm, dtype=np.float32),
            metadata=json.dumps(metadata),
            object_masks_packed=packed_masks,
            object_mask_shape=mask_shape,
        )
        os.replace(tmp, path)
        return path

    def load(self, source_path: str | Path, config: PreprocessConfig) -> ProcessedVolume | None:
        path = self.path_for(source_path, config)
        if not path.exists():
            return None
        with np.load(path, allow_pickle=False) as value:
            label = value["label"]
            image = value["image"].astype(np.float32)
            if image.ndim == 3:
                image = np.repeat(image[None], 3, axis=0)
            return ProcessedVolume(
                image,
                value["valid_mask"].astype(bool),
                None if label.size == 0 else label,
                VolumeTransformRecord.from_dict(json.loads(str(value["transform"]))),
                tuple(float(x) for x in value["spacing"]),
            )

    def load_object_mask(
        self,
        source_path: str | Path,
        config: PreprocessConfig,
        object_key: str,
    ) -> np.ndarray | None:
        path = self.path_for(source_path, config)
        if not path.exists():
            return None
        with np.load(path, allow_pickle=False) as value:
            if "metadata" not in value or "object_masks_packed" not in value:
                return None
            metadata = json.loads(str(value["metadata"]))
            keys = metadata.get("object_keys", [])
            if object_key not in keys:
                return None
            shape = tuple(int(part) for part in value["object_mask_shape"])
            count = int(np.prod(shape))
            index = keys.index(object_key)
            unpacked = np.unpackbits(value["object_masks_packed"][index], count=count)
            return unpacked.reshape(shape).astype(bool)
