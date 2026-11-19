"""PRS-Med data, training, and evaluation command line entry points."""
from __future__ import annotations

import argparse
from datetime import timedelta
from collections import defaultdict
from dataclasses import asdict, replace
from datetime import datetime
from functools import partial
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Iterable

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader

from .config import load_config
from .data.batching import SourceBalancedBucketBatchSampler
from .data.cache import build_cache
from .data.collate import collate_dialogues
from .data.dataset import DatasetConfig, DialogueDataset
from .data.preprocess import PreprocessConfig
from .data.schema import Dialogue
from .data.sources import iter_mrmedseg, iter_our3d
from .data.splits import (
    assert_disjoint,
    drop_mrmedseg_leakage,
    load_split_file,
    overlap_report,
)
from .data.validators import validate_manifest
from .evaluation.metrics import overlap_metrics
from .models.factory import TinyTokenizer, build_model
from .training.trainer import broadcast_trainable_parameters, fit


HEAVY_COMMANDS = {"prepare", "train", "evaluate"}
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _require(path: str, what: str, parser: argparse.ArgumentParser) -> None:
    if not path or not Path(path).exists():
        parser.error(f"mandatory {what} path does not exist: {path!r}")


def _limited(values: Iterable[Dialogue], limit: int | None) -> list[Dialogue]:
    output = []
    for value in values:
        output.append(value)
        if limit is not None and len(output) >= limit:
            break
    return output


def _write_manifest(path: Path, rows: Iterable[Dialogue]) -> int:
    values = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as stream:
        for value in values:
            stream.write(value.to_json() + "\n")
    return len(values)


def _read_manifest(path: str | Path, limit: int | None = None) -> list[Dialogue]:
    output = []
    with Path(path).open() as stream:
        for line in stream:
            if line.strip():
                output.append(Dialogue.from_json(line))
                if limit is not None and len(output) >= limit:
                    break
    return output


def _read_manifests(paths: Iterable[str | Path], limit: int | None = None) -> list[Dialogue]:
    rows: list[Dialogue] = []
    for path in paths:
        rows.extend(_read_manifest(path))
        if limit is not None and len(rows) >= limit:
            return rows[:limit]
    return rows


def _fixed_sample(rows: list[Dialogue], count: int, seed: int) -> list[Dialogue]:
    def key(dialogue: Dialogue) -> str:
        return hashlib.sha256(f"{seed}:{dialogue.dialogue_id}".encode()).hexdigest()

    return sorted(rows, key=key)[:count]


def _source_splits(cfg) -> dict[str, dict[str, set[str]]]:
    output = {}
    for source, path in cfg.sources.get("splits", {}).items():
        if Path(path).is_file():
            output[source] = load_split_file(path, source)
            assert_disjoint(output[source])
    return output


def prepare_manifests(cfg, limit: int | None, requested: list[str] | None) -> dict:
    sources = requested or ["amos", "totalseg", "totalseg_mr", "kits", "mrmedseg"]
    if "dialogues_root" not in cfg.sources or "mrmedseg_root" not in cfg.sources:
        return {
            "overlap": {},
            "mrmedseg_leakage": {"input": 0, "dropped": 0, "kept": 0},
            "counts": {},
        }
    root = Path(cfg.sources["dialogues_root"])
    mr_root = Path(cfg.sources["mrmedseg_root"])
    output_root = Path(cfg.sources.get("manifest_root", "data/manifests"))
    directory_names = {
        "amos": "amos_3d",
        "totalseg": "totalseg_3d",
        "totalseg_mr": "totalseg_mr_3d",
        "kits": "kits_3d",
    }
    all_rows = []
    by_split: dict[str, list[Dialogue]] = defaultdict(list)
    counts: dict[str, int] = {}
    for source in sources:
        if source not in directory_names:
            continue
        for split in ("train", "val", "test", "holdout"):
            source_path = root / directory_names[source] / f"dialogues_{split}.jsonl"
            if not source_path.is_file():
                continue
            rows = _limited(iter_our3d(source_path), limit)
            name = f"{source}_{split}"
            counts[name] = _write_manifest(output_root / f"{name}.jsonl", rows)
            all_rows.extend(rows)
            by_split[split].extend(rows)

    leakage = {"input": 0, "dropped": 0, "kept": 0}
    if "mrmedseg" in sources:
        our_splits = _source_splits(cfg)
        mr_files = {"train": "train.json", "val": "val.json", "test": "test.json"}
        for split, filename in mr_files.items():
            source_path = mr_root / filename
            if not source_path.is_file():
                continue
            # Limit source reads first; this keeps --limit predictably lightweight.
            raw = _limited(iter_mrmedseg(source_path, mr_root, split), limit)
            data_cfg = cfg.data
            policy = (data_cfg.get("mrmedseg_policy") if isinstance(data_cfg, dict)
                      else getattr(data_cfg, "mrmedseg_policy", None)) or "benchmark"
            forbidden = ("val", "test", "holdout") if policy == "strict" else ("val", "test")
            rows, report = drop_mrmedseg_leakage(raw, our_splits, forbidden_splits=forbidden)
            for key in leakage:
                leakage[key] += report[key]
            name = f"mrmedseg_{split}"
            counts[name] = _write_manifest(output_root / f"{name}.jsonl", rows)
            all_rows.extend(rows)
            by_split[split].extend(rows)

    for split, rows in by_split.items():
        counts[split] = _write_manifest(output_root / f"{split}.jsonl", rows)
    report = {
        "overlap": overlap_report(all_rows),
        "mrmedseg_leakage": leakage,
        "counts": counts,
    }
    (output_root / "overlap_report.json").write_text(json.dumps(report, indent=2) + "\n")
    (output_root / "counts.json").write_text(json.dumps(counts, indent=2) + "\n")
    return report


def split_manifests(cfg) -> dict:
    output_root = Path(cfg.sources.get("manifest_root", "data/manifests"))
    rows = []
    for path in output_root.glob("*_*.jsonl"):
        if path.stem in {"train", "val", "test", "holdout"}:
            continue
        rows.extend(_read_manifest(path))
    validate_manifest(rows)
    report = overlap_report(rows)
    if any(report.values()):
        raise ValueError(f"patient overlap remains after split derivation: {report}")
    (output_root / "split_report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def audit_manifest(path: str | Path, output: str | Path | None = None) -> dict:
    rows = _read_manifest(path)
    validate_manifest(rows)
    summary = {
        "manifest": str(path),
        "dialogues": len(rows),
        "turns": sum(len(row.turns) for row in rows),
        "actions": sum(len(turn.actions) for row in rows for turn in row.turns),
        "sources": dict(sorted(_counts(row.source for row in rows).items())),
        "splits": dict(sorted(_counts(row.split for row in rows).items())),
    }
    destination = Path(output) if output else Path(path).with_suffix(".audit.json")
    destination.write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def _counts(values: Iterable[str]) -> dict[str, int]:
    output: dict[str, int] = defaultdict(int)
    for value in values:
        output[value] += 1
    return dict(output)


def _dataset_config(cfg, model, tiny: bool) -> DatasetConfig:
    if tiny:
        global_budget = model.global_pooler.budget
        detail_tokens = model.query_encoder.slots.shape[0]
        region_tokens = model.region_tokenizer.tokens_per_entity
    else:
        global_budget = int(cfg.model.get("global", {}).get("budget", 512))
        detail_tokens = int(cfg.model.get("detail", {}).get("K", 64))
        region_tokens = int(cfg.model.get("region", {}).get("tokens_per_entity", 4))
    return DatasetConfig(
        max_len=cfg.runtime.max_len,
        image_size=cfg.data.image_size,
        d_max=cfg.data.d_max,
        global_budget=global_budget,
        detail_tokens=detail_tokens,
        region_tokens_per_entity=region_tokens,
        seed=cfg.runtime.seed,
        processed_root=cfg.data.processed_root,
        target_inplane_spacing_mm=cfg.data.target_inplane_spacing_mm,
        target_depth_spacing_mm=cfg.data.target_depth_spacing_mm,
        max_native_depth_spacing_mm=cfg.data.max_native_depth_spacing_mm,
        feature_root=cfg.data.feature_root,
    )


def _loader(rows, tokenizer, dataset_cfg, cfg, train: bool):
    dataset = DialogueDataset(rows, tokenizer, dataset_cfg, mode="train" if train else "eval")
    batch_size = int(cfg.train.get("batch_size", 1))
    sampler = SourceBalancedBucketBatchSampler(
        rows,
        batch_size=batch_size,
        seed=cfg.runtime.seed,
        balance_by_turns=bool(cfg.train.get("balance_by_turns", False)),
        match_ranks=bool(cfg.train.get("match_rank_work", False)) and train,
    )
    loader_args = {
        "dataset": dataset,
        "batch_sampler": sampler,
        "num_workers": cfg.runtime.num_workers,
        "pin_memory": cfg.runtime.pin_memory,
        "collate_fn": partial(collate_dialogues, pad_token_id=tokenizer.pad_token_id or 0),
    }
    if cfg.runtime.num_workers > 0:
        loader_args["persistent_workers"] = cfg.runtime.persistent_workers
        loader_args["prefetch_factor"] = cfg.runtime.prefetch_factor
    return DataLoader(
        **loader_args,
    )


def _distributed_context() -> tuple[int, int, int]:
    return (
        int(os.environ.get("RANK", "0")),
        int(os.environ.get("WORLD_SIZE", "1")),
        int(os.environ.get("LOCAL_RANK", "0")),
    )


def _init_distributed() -> tuple[int, int, int]:
    rank, world_size, local_rank = _distributed_context()
    if world_size > 1 and not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        # Bounded timeout: if one rank raises, the others fail within minutes instead of hanging.
        timeout_minutes = float(os.environ.get("PRSMED15_DIST_TIMEOUT_MIN", "20"))
        dist.init_process_group(
            backend=backend,
            init_method="env://",
            timeout=timedelta(minutes=timeout_minutes),
        )
    return rank, world_size, local_rank


def _resolve_run_paths(cfg, results_override: str | None) -> tuple[Path, Path]:
    run_name = str(cfg.train.get("run_name", f"stage_{cfg.train.get('stage', 'train')}"))
    if results_override:
        results = Path(results_override)
    elif cfg.train.get("results_dir"):
        results = Path(cfg.train["results_dir"])
    else:
        root = Path(
            cfg.train.get(
                "results_root",
                "/data2/gdp2988/prs_med/knowledge/experimental_results",
            )
        )
        results = root / f"{datetime.now().date().isoformat()}_{run_name}"
    checkpoint_value = cfg.train.get("checkpoint_dir")
    checkpoints = Path(checkpoint_value or f"artifacts/runs/{run_name}")
    cfg.train["results_dir"] = str(results)
    if checkpoint_value or not results_override:
        cfg.train["checkpoint_dir"] = str(checkpoints)
    return results, checkpoints


def _write_run_readme(cfg, manifests: list[str], results: Path) -> None:
    results.mkdir(parents=True, exist_ok=True)
    raw = asdict(cfg)
    config_hash = hashlib.sha256(json.dumps(raw, sort_keys=True, default=str).encode()).hexdigest()
    manifest_hashes = {
        path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
        for path in manifests
    }
    hardware = {
        "world_size": _distributed_context()[1],
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "gpu_names": [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())],
    }
    lines = [
        f"# {cfg.train.get('run_name', 'PRS-Med15 run')}",
        "",
        "Auto-generated run stub. Add experiment notes and conclusions here.",
        "",
        f"- Config SHA256: `{config_hash}`",
        f"- Seed: `{cfg.runtime.seed}`",
        f"- Hardware: `{json.dumps(hardware, sort_keys=True)}`",
        f"- Manifest SHA256: `{json.dumps(manifest_hashes, sort_keys=True)}`",
        "",
    ]
    (results / "README.md").write_text("\n".join(lines))
    metrics = results / "metrics.json"
    if not metrics.exists():
        metrics.write_text("[]\n")


def train_command(
    cfg,
    tiny: bool,
    limit: int | None,
    results_dir: str | None,
    init_from: str | None = None,
):
    rank, world_size, local_rank = _init_distributed()
    if not cfg.model:
        cfg.model = load_config(PROJECT_ROOT / "configs/model/shared_slice.yaml").model
    model, tokenizer, _ = build_model(cfg, tiny=tiny)
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")
    model = model.to(device)
    if world_size > 1:
        # No DDP wrapper: the forward runs the LLM / decoder / region modules a data-dependent
        # number of times, and DDP(find_unused_parameters=True) + non-reentrant checkpointing
        # desynchronised bucket all-reduces between ranks (NCCL hang, 2026-11-16). Instead the
        # trainer averages all trainable gradients with one flat all-reduce per optimizer step
        # (same math, identical collective sequence on every rank). Start from identical weights:
        broadcast_trainable_parameters(model)
    if rank == 0:
        print(f"[train] device={device} world_size={world_size}", flush=True)
    _resolve_run_paths(cfg, results_dir)
    if init_from:
        cfg.train["init_from"] = init_from
    train_paths = list(cfg.data.manifests) or [cfg.data.manifest]
    train_rows = _read_manifests(train_paths, limit)
    val_paths = list(cfg.data.val_manifests)
    val_path = cfg.train.get("val_manifest")
    if not val_paths and val_path:
        val_paths = [val_path]
    val_rows = _read_manifests(val_paths) if val_paths else []
    val_limit = int(cfg.train.get("val_sample_size", 0))
    if val_limit:
        val_rows = _fixed_sample(val_rows, val_limit, cfg.runtime.seed)
    if limit is not None:
        val_rows = val_rows[:limit]
    if rank == 0:
        _write_run_readme(cfg, train_paths + val_paths, Path(cfg.train["results_dir"]))
    if dist.is_initialized():
        dist.barrier()
    dataset_cfg = _dataset_config(cfg, model, tiny)
    train_loader = _loader(train_rows, tokenizer, dataset_cfg, cfg, train=True)
    val_loader = _loader(val_rows, tokenizer, dataset_cfg, cfg, train=False) if val_rows else None
    return fit(model, train_loader, val_loader, cfg)


def cache_command(cfg, requested: list[str] | None, workers: int | None, limit: int | None):
    manifest_root = Path(cfg.sources.get("manifest_root", "data/manifests"))
    source_names = requested or ["amos", "totalseg", "totalseg_mr", "kits"]
    manifests = [
        manifest_root / f"{source}_{split}.jsonl"
        for source in source_names
        for split in ("train", "val", "test", "holdout")
        if (manifest_root / f"{source}_{split}.jsonl").is_file()
    ]
    preprocess_config = PreprocessConfig(
        image_size=cfg.data.image_size,
        target_inplane_spacing_mm=cfg.data.target_inplane_spacing_mm,
        target_depth_spacing_mm=cfg.data.target_depth_spacing_mm,
        max_native_depth_spacing_mm=cfg.data.max_native_depth_spacing_mm,
        d_max=cfg.data.d_max,
    )
    return build_cache(
        manifests,
        cfg.data.processed_root,
        preprocess_config,
        set(source_names),
        workers,
        limit,
        max_worker_memory_gb=float(os.environ.get("PRSMED15_CACHE_WORKER_GB", "18")),
    )


def profile_loader(cfg, limit: int) -> dict[str, float | int]:
    paths = list(cfg.data.manifests) or [cfg.data.manifest]
    rows = _read_manifests(paths, limit)
    base = DatasetConfig(
        max_len=cfg.runtime.max_len,
        image_size=cfg.data.image_size,
        d_max=cfg.data.d_max,
        processed_root=cfg.data.processed_root,
        target_inplane_spacing_mm=cfg.data.target_inplane_spacing_mm,
        target_depth_spacing_mm=cfg.data.target_depth_spacing_mm,
        max_native_depth_spacing_mm=cfg.data.max_native_depth_spacing_mm,
        global_budget=16,
        detail_tokens=4,
        region_tokens_per_entity=4,
    )
    tokenizer = TinyTokenizer()
    timings: dict[str, float | int] = {"items": len(rows)}
    for label, use_cache in (("cached", True), ("uncached", False)):
        selected = replace(base, use_cache=use_cache)
        dataset = DialogueDataset(rows, tokenizer, selected, mode="eval")
        started = time.perf_counter()
        for index in range(len(dataset)):
            dataset[index]
        elapsed = time.perf_counter() - started
        timings[f"{label}_mean_seconds_per_item"] = elapsed / max(1, len(dataset))
    print(json.dumps(timings, indent=2))
    return timings


@torch.no_grad()
def evaluate_command(cfg, tiny: bool, history: str, limit: int | None, results_dir: str | None):
    if not cfg.model:
        cfg.model = load_config(PROJECT_ROOT / "configs/model/shared_slice.yaml").model
    model, tokenizer, _ = build_model(cfg, tiny=tiny)
    model.history_mode = history
    manifest = cfg.eval.get("manifest", cfg.data.manifest)
    rows = _read_manifest(manifest, limit)
    loader = _loader(rows, tokenizer, _dataset_config(cfg, model, tiny), cfg, train=False)
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    valid_actions = []
    for batch in loader:
        output = model(batch)
        for action in output.actions:
            target = action.get("target")
            valid_actions.append(target is not None)
            if target is None:
                continue
            metrics = overlap_metrics(action["logits"][:, 0], target)
            grouped[(action["dimension"], action["source"])].append(metrics)
    result = {"history": history, "groups": {}, "invalid_action_rate": 0.0}
    for (dimension, source), values in grouped.items():
        result["groups"][f"{dimension}:{source}"] = {
            "actions": len(values),
            "dice": sum(value["dice"] for value in values) / len(values),
            "iou": sum(value["iou"] for value in values) / len(values),
        }
    if valid_actions:
        result["invalid_action_rate"] = 1 - sum(valid_actions) / len(valid_actions)
    destination = Path(results_dir or cfg.eval.get("results_dir", "artifacts/evaluation"))
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="prs-med15")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare", "split", "audit", "cache", "train", "evaluate", "chat", "profile"):
        command = sub.add_parser(name)
        command.add_argument("--config", default="configs/base.yaml")
        command.add_argument("--set", action="append", default=[])
        if name in HEAVY_COMMANDS:
            command.add_argument("--limit", type=int)
        if name == "prepare":
            command.add_argument("--sources", nargs="+")
        if name == "cache":
            command.add_argument("--sources", nargs="+")
            command.add_argument("--workers", type=int)
            command.add_argument("--limit", type=int)
        if name in {"audit", "split"}:
            command.add_argument("--manifest")
        if name in {"train", "evaluate"}:
            command.add_argument("--results_dir")
            command.add_argument("--tiny", action="store_true")
        if name == "train":
            command.add_argument("--init_from")
        if name == "evaluate":
            command.add_argument("--history", choices=("oracle", "predicted"), default="oracle")
        if name == "profile":
            command.add_argument("--what", choices=("loader",), required=True)
            command.add_argument("--limit", type=int, default=20)
    args = parser.parse_args(argv)
    _require(args.config, "config", parser)
    cfg = load_config(args.config, args.set)
    if (
        args.command in HEAVY_COMMANDS
        and getattr(args, "limit", None) is None
        and os.environ.get("PRSMED15_ALLOW_HEAVY") != "1"
    ):
        parser.error(f"{args.command} without --limit requires PRSMED15_ALLOW_HEAVY=1")

    if args.command == "prepare":
        print(json.dumps(prepare_manifests(cfg, args.limit, args.sources), indent=2))
        return 0
    if args.command == "split":
        print(json.dumps(split_manifests(cfg), indent=2))
        return 0
    if args.command == "audit":
        manifest = args.manifest or cfg.data.manifest
        _require(manifest, "manifest", parser)
        print(json.dumps(audit_manifest(manifest), indent=2))
        return 0
    if args.command == "cache":
        result = cache_command(cfg, args.sources, args.workers, args.limit)
        print(json.dumps({key: value for key, value in result.items() if key != "items"}, indent=2))
        return 0
    if args.command == "train":
        train_manifests = list(cfg.data.manifests) or [cfg.data.manifest]
        for manifest in train_manifests:
            _require(manifest, "manifest", parser)
        try:
            result = train_command(
                cfg,
                args.tiny,
                args.limit,
                args.results_dir,
                args.init_from,
            )
            if not dist.is_initialized() or dist.get_rank() == 0:
                print(json.dumps(result, indent=2))
        finally:
            if dist.is_initialized():
                dist.destroy_process_group()
        return 0
    if args.command == "evaluate":
        manifest = cfg.eval.get("manifest", cfg.data.manifest)
        _require(manifest, "manifest", parser)
        result = evaluate_command(
            cfg,
            args.tiny,
            args.history,
            args.limit,
            args.results_dir,
        )
        print(json.dumps(result, indent=2))
        return 0
    if args.command == "profile":
        profile_loader(cfg, args.limit)
        return 0
    print(f"{args.command}: NotImplemented (runtime inference is Phase 3)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
