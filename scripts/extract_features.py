"""Precompute frozen MedSAM slice features for every train scan (resumable, sharded).

Usage (one process per GPU):
  CUDA_VISIBLE_DEVICES=3 PRSMED15_ALLOW_HEAVY=1 PYTHONPATH=src \
    python scripts/extract_features.py --shard 0 --num-shards 2 [--limit N] [--workers 2]

Features are produced by the exact training encoder module on the exact tensors the dataset
loads, under the same bf16 autocast, and stored as float16 [D_full, 32, 32, 256].
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from prs_med15.cli import PROJECT_ROOT, _dataset_config, _read_manifest
from prs_med15.config import load_config
from prs_med15.data.dataset import _default_scan_loader
from prs_med15.data.features import feature_path, save_features
from prs_med15.models.factory import _model_config

MANIFESTS = ["amos_train", "totalseg_train", "totalseg_mr_train", "kits_train", "mrmedseg_train"]


class ScanSet(Dataset):
    def __init__(self, scans, cfg, root):
        self.scans, self.cfg, self.root = scans, cfg, root

    def __len__(self):
        return len(self.scans)

    def __getitem__(self, i):
        source, scan = self.scans[i]
        try:
            image, valid, geo = _default_scan_loader(scan, self.cfg, source)
        except Exception as exc:  # report, never kill the whole extraction
            return {"error": f"{scan.image_path} :: {exc!r}"}
        path = feature_path(self.root, source, scan.image_path, geo if isinstance(geo, dict) else None)
        return {"image": image.float(), "valid": valid.bool(), "path": str(path)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--root", default="data/features")
    ap.add_argument("--slices-per-pass", type=int, default=32)
    ap.add_argument("--kind", choices=["all", "volume", "native_2d"], default="all")
    args = ap.parse_args()

    cfg = load_config("configs/train/full_stage_b.yaml", [])
    cfg.model = load_config(PROJECT_ROOT / "configs/model/shared_slice.yaml").model
    dataset_cfg = _dataset_config(cfg, None, False)

    seen, scans = set(), []
    for name in MANIFESTS:
        for row in _read_manifest(f"data/manifests/{name}.jsonl"):
            for scan in row.scans:
                key = (row.source, scan.image_path)
                if key not in seen:
                    seen.add(key)
                    scans.append((row.source, scan))
    if args.kind != "all":
        scans = [value for value in scans if (value[1].input_kind == "native_2d") == (args.kind == "native_2d")]
    # Volumes first (largest, most reused); deterministic order so shards are stable.
    scans.sort(key=lambda value: (value[1].input_kind == "native_2d", value[0], str(value[1].image_path)))
    scans = scans[args.shard::args.num_shards]
    if args.limit:
        scans = scans[:args.limit]

    from prs_med15.models.encoder import MedSamSliceEncoder
    enc_cfg = _model_config(cfg)["encoder"]
    device = torch.device("cuda")
    encoder = MedSamSliceEncoder(enc_cfg.get("hf_id", "wanglab/medsam-vit-base"),
                                 image_size=int(enc_cfg.get("image_size", 512)),
                                 microbatch=int(enc_cfg.get("slice_microbatch", 8)),
                                 freeze=True, cache_dir=enc_cfg["hf_cache"],
                                 local_files_only=True).to(device).eval()

    root = Path(args.root)
    # Skip finished files cheaply for volumes (key known from cache path) by resolving in workers.
    loader = DataLoader(ScanSet(scans, dataset_cfg, root), batch_size=None,
                        num_workers=args.workers, prefetch_factor=2 if args.workers else None)
    started, done, slices, skipped, errors = time.time(), 0, 0, 0, 0
    for item in loader:
        if "error" in item:
            errors += 1
            print(json.dumps({"error": item["error"]}), file=sys.stderr, flush=True)
            continue
        path = Path(item["path"])
        if path.exists():
            skipped += 1
            continue
        image, valid = item["image"], item["valid"]  # [C,D,H,W], [D,H,W]
        out = []
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            for z in range(0, image.shape[1], args.slices_per_pass):
                chunk = image[None, :, z:z + args.slices_per_pass].to(device, non_blocking=True)
                vchunk = valid[None, z:z + args.slices_per_pass].to(device, non_blocking=True)
                out.append(encoder(chunk, vchunk).dense[0].to(torch.float16).cpu())
        save_features(path, torch.cat(out).numpy())
        done += 1
        slices += image.shape[1]
        if done % 50 == 0:
            rate = slices / (time.time() - started)
            print(json.dumps({"shard": args.shard, "done": done, "skipped": skipped, "errors": errors,
                              "of": len(scans), "slices": slices, "slices_per_s": round(rate, 1)}),
                  flush=True)
    print(json.dumps({"shard": args.shard, "finished": True, "done": done, "skipped": skipped,
                      "errors": errors, "of": len(scans), "slices": slices,
                      "seconds": round(time.time() - started)}), flush=True)


if __name__ == "__main__":
    main()
