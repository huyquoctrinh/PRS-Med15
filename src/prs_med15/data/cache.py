"""Build and inspect the resumable processed-volume cache."""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Iterable

import nibabel as nib
import numpy as np

from .geometry import apply_volume_transform
from .preprocess import PreprocessConfig, ProcessedCache, preprocess_volume
from .schema import Dialogue


@dataclass
class CacheJob:
    source: str
    image_path: str
    modality: str
    label_ref: str | None
    object_masks: dict[str, str]


def _canonical_dhw(path: str, dtype=np.float32) -> tuple[np.ndarray, nib.Nifti1Image, nib.Nifti1Image]:
    """Load in RAS order as [D,H,W]; float32 (not nibabel's default float64) to bound memory."""
    source = nib.load(path)
    canonical = nib.as_closest_canonical(source)
    xyz = canonical.get_fdata(dtype=dtype, caching="unchanged")
    return np.transpose(xyz, (2, 1, 0)), source, canonical


def _limit_worker_memory(max_bytes: int | None) -> None:
    """Cap a worker's address space so a huge volume raises MemoryError for that job only,
    instead of driving the shared machine into the kernel OOM killer."""
    if not max_bytes:
        return
    import resource
    resource.setrlimit(resource.RLIMIT_AS, (int(max_bytes), int(max_bytes)))


def _mask_path(label_ref: str | None, mask_file: str | None, mask_path: str | None) -> str | None:
    if mask_path and Path(mask_path).name.endswith((".nii", ".nii.gz")):
        return mask_path
    if label_ref and mask_file:
        candidate = Path(label_ref) / mask_file
        if candidate.is_file():
            return str(candidate)
    return None


def _mask_key(mask_file: str | None, mask_path: str | None, object_key: str) -> str:
    """Use the annotation identity; dialogue-local object keys are often reused."""
    return mask_file or mask_path or object_key


def collect_cache_jobs(
    manifest_paths: Iterable[str | Path],
    requested_sources: set[str] | None = None,
    limit: int | None = None,
) -> list[CacheJob]:
    """Collect unique 3D scans and the union of masks referenced by dialogues."""
    jobs: dict[str, CacheJob] = {}
    for manifest_path in manifest_paths:
        with Path(manifest_path).open() as stream:
            for line in stream:
                if not line.strip():
                    continue
                dialogue = Dialogue.from_json(line)
                if requested_sources and dialogue.source not in requested_sources:
                    continue
                for scan in dialogue.scans:
                    if scan.input_kind == "native_2d":
                        continue
                    resolved = str(Path(scan.image_path).resolve())
                    if limit is not None and resolved not in jobs and len(jobs) >= limit:
                        return list(jobs.values())
                    job = jobs.setdefault(
                        resolved,
                        CacheJob(
                            source=dialogue.source,
                            image_path=scan.image_path,
                            modality=scan.modality,
                            label_ref=scan.label_ref,
                            object_masks={},
                        ),
                    )
                    for obj in dialogue.objects.values():
                        path = _mask_path(scan.label_ref, obj.mask_file, obj.mask_path)
                        if path:
                            key = _mask_key(obj.mask_file, obj.mask_path, obj.object_key)
                            job.object_masks[key] = path
    return list(jobs.values())


def _build_one(job: CacheJob, processed_root: str, config_values: dict) -> dict:
    config = PreprocessConfig(**config_values)
    cache = ProcessedCache(Path(processed_root) / job.source)
    destination = cache.path_for(job.image_path, config)
    if destination.exists():
        return {"status": "skipped", "source": job.source, "image_path": job.image_path,
                "cache_path": str(destination)}

    image, source_nii, canonical_nii = _canonical_dhw(job.image_path)
    spacing = tuple(reversed(canonical_nii.header.get_zooms()[:3]))
    label = None
    if job.label_ref and Path(job.label_ref).is_file():
        label, _, _ = _canonical_dhw(job.label_ref)
        label = np.rint(label).astype(np.int16)
    processed = preprocess_volume(
        image,
        spacing,
        job.modality,
        config,
        label=label,
        source_affine=source_nii.affine,
        canonical_affine=canonical_nii.affine,
        replicate_channels=False,
    )
    del image, label

    packed_masks: dict[str, np.ndarray] = {}
    if job.object_masks:
        merged = np.zeros(processed.valid_mask.shape, dtype=np.int16)
        for value, (object_key, path) in enumerate(sorted(job.object_masks.items()), start=1):
            raw_mask, _, _ = _canonical_dhw(path)
            raw_mask = raw_mask > 0
            mask = apply_volume_transform(raw_mask, processed.transform, order=0).astype(bool)
            del raw_mask
            packed_masks[object_key] = np.packbits(mask.reshape(-1))
            merged[mask] = value
            del mask
        processed.label = merged
    cache.save(job.image_path, config, processed, packed_object_masks=packed_masks)
    return {"status": "built", "source": job.source, "image_path": job.image_path,
            "cache_path": str(destination)}


def build_cache(
    manifest_paths: Iterable[str | Path],
    processed_root: str | Path,
    config: PreprocessConfig,
    requested_sources: set[str] | None = None,
    workers: int | None = None,
    limit: int | None = None,
    max_worker_memory_gb: float | None = 24.0,
) -> dict:
    jobs = collect_cache_jobs(manifest_paths, requested_sources, limit)
    worker_count = workers or min(6, max(1, (os.cpu_count() or 2) // 2))
    memory_cap = int(max_worker_memory_gb * 2**30) if max_worker_memory_gb else None
    report = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "processed_root": str(processed_root),
        "preprocess_config": asdict(config),
        "workers": worker_count,
        "total": len(jobs),
        "built": 0,
        "skipped": 0,
        "failed": 0,
        "items": [],
    }
    Path(processed_root).mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(
        max_workers=worker_count,
        initializer=_limit_worker_memory,
        initargs=(memory_cap,),
    ) as executor:
        futures = {
            executor.submit(_build_one, job, str(processed_root), asdict(config)): job
            for job in jobs
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            job = futures[future]
            try:
                item = future.result()
            except Exception as error:
                item = {
                    "status": "failed",
                    "source": job.source,
                    "image_path": job.image_path,
                    "error": f"{type(error).__name__}: {error}",
                }
            report[item["status"]] += 1
            report["items"].append(item)
            suffix = f" :: {item['error']}" if item.get("error") else ""
            print(
                f"[cache] {completed}/{len(jobs)} {item['status']} {job.image_path}{suffix}",
                flush=True,
            )
    report["finished_at"] = datetime.now(timezone.utc).isoformat()
    report_path = Path(processed_root) / "cache_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    report["report_path"] = str(report_path)
    return report
