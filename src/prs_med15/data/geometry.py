"""RAS canonicalisation and explicit forward/inverse spatial transforms."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import nibabel as nib
import numpy as np
from scipy import ndimage


def load_canonical(image_path: str, label_path: str | None = None) -> dict[str, Any]:
    """Load an image and optional label in RAS+ voxel order.

    Adapted from PRS-Med/data_processing/anatomy/canonicalise.py.
    """
    image = nib.load(image_path)
    canonical = nib.as_closest_canonical(image)
    out: dict[str, Any] = {
        "image": np.asarray(canonical.dataobj), "affine": canonical.affine.copy(),
        "spacing": tuple(float(x) for x in canonical.header.get_zooms()[:3]),
        "src_axcodes": "".join(nib.aff2axcodes(image.affine)),
    }
    if label_path is not None:
        label = nib.as_closest_canonical(nib.load(label_path))
        if label.shape != canonical.shape:
            raise ValueError("canonical image and label shapes differ")
        out["label"] = np.asarray(label.dataobj)
    return out


@dataclass(frozen=True)
class TransformRecord:
    original_shape: tuple[int, int]
    resized_shape: tuple[int, int]
    output_shape: tuple[int, int]
    scale: tuple[float, float]
    pad: tuple[int, int, int, int]  # top, bottom, left, right
    unknown_geometry: bool = False

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["kind"] = "inplane"
        return value

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "TransformRecord":
        value = dict(value)
        value.pop("kind", None)
        return cls(**{k: tuple(v) if isinstance(v, list) else v for k, v in value.items()})


@dataclass(frozen=True)
class VolumeTransformRecord:
    """Complete raw [D,H,W] to processed transform, including full-FOV crop metadata."""
    original_shape: tuple[int, int, int]
    resampled_shape: tuple[int, int, int]
    output_shape: tuple[int, int, int]
    zoom: tuple[float, float, float]
    crop: tuple[tuple[int, int], tuple[int, int], tuple[int, int]]
    letterbox: TransformRecord
    source_affine: tuple[tuple[float, ...], ...] | None = None
    canonical_affine: tuple[tuple[float, ...], ...] | None = None
    unknown_geometry: bool = False

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["kind"] = "volume"
        return value

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "VolumeTransformRecord":
        value = dict(value)
        value.pop("kind", None)
        value["original_shape"] = tuple(value["original_shape"])
        value["resampled_shape"] = tuple(value["resampled_shape"])
        value["output_shape"] = tuple(value["output_shape"])
        value["zoom"] = tuple(value["zoom"])
        value["crop"] = tuple(tuple(x) for x in value["crop"])
        value["letterbox"] = TransformRecord.from_dict(value["letterbox"])
        for key in ("source_affine", "canonical_affine"):
            if value.get(key) is not None:
                value[key] = tuple(tuple(row) for row in value[key])
        return cls(**value)


def make_letterbox_transform(shape: tuple[int, int], output_shape: tuple[int, int] = (512, 512),
                             scale: float | None = None) -> TransformRecord:
    h, w = shape
    factor = min(output_shape[0] / h, output_shape[1] / w) if scale is None else scale
    rh, rw = max(1, round(h * factor)), max(1, round(w * factor))
    if rh > output_shape[0] or rw > output_shape[1]:
        raise ValueError("resized image does not fit output")
    top = (output_shape[0] - rh) // 2
    left = (output_shape[1] - rw) // 2
    return TransformRecord(shape, (rh, rw), output_shape, (rh / h, rw / w),
                           (top, output_shape[0] - rh - top, left, output_shape[1] - rw - left))


def apply_transform(array: np.ndarray, transform: TransformRecord, *, order: int = 0) -> np.ndarray:
    if array.shape[-2:] != transform.original_shape:
        raise ValueError("input shape does not match transform")
    zoom = (1,) * (array.ndim - 2) + transform.scale
    resized = ndimage.zoom(array, zoom, order=order, mode="nearest", prefilter=order > 1)
    # scipy rounding can differ by one; crop/pad to the recorded shape.
    resized = resized[..., :transform.resized_shape[0], :transform.resized_shape[1]]
    t, b, l, r = transform.pad
    return np.pad(resized, ((0, 0),) * (array.ndim - 2) + ((t, b), (l, r)))


def invert_transform(array: np.ndarray, transform: TransformRecord, *, order: int = 0) -> np.ndarray:
    if array.shape[-2:] != transform.output_shape:
        raise ValueError("input shape does not match transform output")
    t, _, l, _ = transform.pad
    rh, rw = transform.resized_shape
    cropped = array[..., t:t + rh, l:l + rw]
    zoom = (1,) * (array.ndim - 2) + (transform.original_shape[0] / rh, transform.original_shape[1] / rw)
    restored = ndimage.zoom(cropped, zoom, order=order, mode="nearest", prefilter=order > 1)
    return restored[..., :transform.original_shape[0], :transform.original_shape[1]]


def apply_volume_transform(array: np.ndarray, transform: VolumeTransformRecord, *, order: int = 0) -> np.ndarray:
    """Apply the recorded crop/resample/letterbox path to a raw-space volume mask or image."""
    if tuple(array.shape[-3:]) != transform.original_shape:
        raise ValueError("raw volume shape does not match transform")
    prefix = (1,) * (array.ndim - 3)
    resized = ndimage.zoom(array, prefix + transform.zoom, order=order, mode="nearest", prefilter=order > 1)
    resized = resized[..., :transform.resampled_shape[0], :transform.resampled_shape[1], :transform.resampled_shape[2]]
    return apply_transform(resized, transform.letterbox, order=order)


def invert_volume_transform(array: np.ndarray, transform: VolumeTransformRecord, *, order: int = 0) -> np.ndarray:
    """Map a processed volume back to the recorded original voxel grid."""
    if tuple(array.shape[-3:]) != transform.output_shape:
        raise ValueError("processed volume shape does not match transform")
    unboxed = invert_transform(array, transform.letterbox, order=order)
    prefix = (1,) * (array.ndim - 3)
    inverse_zoom = tuple(o / r for o, r in zip(transform.original_shape, transform.resampled_shape, strict=True))
    raw = ndimage.zoom(unboxed, prefix + inverse_zoom, order=order, mode="nearest", prefilter=order > 1)
    return raw[..., :transform.original_shape[0], :transform.original_shape[1], :transform.original_shape[2]]


def voxel_to_world(affine: np.ndarray, coordinates: np.ndarray) -> np.ndarray:
    return nib.affines.apply_affine(affine, coordinates)


def world_to_voxel(affine: np.ndarray, coordinates: np.ndarray) -> np.ndarray:
    return nib.affines.apply_affine(np.linalg.inv(affine), coordinates)


def physical_grid(affine: np.ndarray | None, grid_shape: tuple[int, int, int],
                  voxel_shape: tuple[int, int, int]) -> tuple[np.ndarray, np.ndarray, bool]:
    """Return RAS cell centres/extents in mm, or zeros plus unknown flag."""
    if affine is None:
        return np.zeros((*grid_shape, 3), np.float32), np.zeros((*grid_shape, 3), np.float32), True
    steps = np.asarray(voxel_shape, float) / np.asarray(grid_shape, float)
    axes = [(np.arange(n) + .5) * step - .5 for n, step in zip(grid_shape, steps, strict=True)]
    ijk = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)
    centers = nib.affines.apply_affine(affine, ijk).astype(np.float32)
    voxel_mm = np.linalg.norm(np.asarray(affine)[:3, :3], axis=0)
    extents = np.broadcast_to(voxel_mm * steps, centers.shape).astype(np.float32).copy()
    return centers, extents, False
