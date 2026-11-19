"""Optimization, validation, and component-wise resumable checkpoints."""
from __future__ import annotations

from contextlib import nullcontext
from dataclasses import asdict, is_dataclass
import hashlib
import json
import sys
import os
import time
import math
from pathlib import Path
import random
from typing import Any

import numpy as np
import torch
import torch.distributed as dist

from .losses import language_loss, total_loss


def _distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


def _rank() -> int:
    return dist.get_rank() if _distributed() else 0


def _unwrap(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if hasattr(model, "module") else model


def _collective_all(value: bool, device: torch.device) -> bool:
    flag = torch.tensor(int(value), device=device, dtype=torch.int32)
    if _distributed():
        dist.all_reduce(flag, op=dist.ReduceOp.MIN)
    return bool(flag.item())


def _trainable_parameters(model: torch.nn.Module) -> list[torch.nn.Parameter]:
    return [parameter for parameter in model.parameters() if parameter.requires_grad]


@torch.no_grad()
def broadcast_trainable_parameters(model: torch.nn.Module, source: int = 0) -> None:
    """Make every rank start from rank ``source``'s trainable weights (frozen weights are loaded
    from the same checkpoint on every rank)."""
    if not _distributed():
        return
    for parameter in _trainable_parameters(model):
        dist.broadcast(parameter.data, src=source)


@torch.no_grad()
def average_gradients(model: torch.nn.Module, bucket_elements: int = 32_000_000) -> None:
    """Average trainable gradients across ranks with a fixed, rank-independent collective order.

    Missing gradients (parameter unused on this rank this step) count as zeros, so every rank
    issues exactly the same all-reduces regardless of which modules its batch touched.
    """
    if not _distributed():
        return
    world = dist.get_world_size()
    by_dtype: dict[torch.dtype, list[torch.nn.Parameter]] = {}
    for parameter in _trainable_parameters(model):
        if parameter.grad is None:
            parameter.grad = torch.zeros_like(parameter)
        by_dtype.setdefault(parameter.grad.dtype, []).append(parameter)
    for dtype in sorted(by_dtype, key=str):
        bucket: list[torch.nn.Parameter] = []
        size = 0
        for parameter in by_dtype[dtype] + [None]:
            if parameter is not None:
                bucket.append(parameter)
                size += parameter.numel()
            if bucket and (parameter is None or size >= bucket_elements):
                flat = torch.cat([item.grad.reshape(-1) for item in bucket])
                dist.all_reduce(flat, op=dist.ReduceOp.SUM)
                flat /= world
                offset = 0
                for item in bucket:
                    count = item.numel()
                    item.grad.copy_(flat[offset:offset + count].view_as(item.grad))
                    offset += count
                bucket, size = [], 0


def parameter_groups(
    model: torch.nn.Module,
    new_lr: float,
    encoder_lr: float = 0.0,
    lora_lr: float | None = None,
    delta_lr: float | None = None,
):
    buckets = {"encoder": [], "lora": [], "delta": [], "new": []}
    seen: set[int] = set()
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad or id(parameter) in seen:
            continue
        seen.add(id(parameter))
        normalized_name = name.removeprefix("module.")
        if normalized_name.startswith("encoder."):
            key = "encoder"
        elif "lora_" in normalized_name:
            key = "lora"
        elif ".delta" in normalized_name:
            key = "delta"
        else:
            key = "new"
        buckets[key].append(parameter)
    learning_rates = {
        "encoder": encoder_lr,
        "lora": lora_lr or new_lr,
        "delta": delta_lr or new_lr,
        "new": new_lr,
    }
    groups = [
        {"params": parameters, "lr": learning_rates[name], "name": name}
        for name, parameters in buckets.items()
        if parameters and learning_rates[name] > 0
    ]
    counts = {
        name: sum(parameter.numel() for parameter in parameters)
        for name, parameters in buckets.items()
    }
    return groups, counts


def cosine_scheduler(optimizer, warmup_steps: int, total_steps: int):
    def scale(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * min(progress, 1)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, scale)


def _rng_state():
    return {
        "torch": torch.get_rng_state(),
        "numpy": np.random.get_state(),
        "python": random.getstate(),
    }


def _restore_rng(state: dict[str, Any]) -> None:
    torch.set_rng_state(state["torch"])
    np.random.set_state(state["numpy"])
    random.setstate(state["python"])


def save_checkpoint(path: str | Path, model, optimizer, scheduler, step: int, extra=None):
    """Legacy single-file helper retained for Phase 2 checkpoint compatibility."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    trainable = {
        name: parameter.detach().cpu()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    torch.save(
        {
            "model": trainable,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "step": step,
            "rng": _rng_state(),
            "extra": extra or {},
        },
        destination,
    )


def load_checkpoint(path: str | Path, model, optimizer=None, scheduler=None):
    state = torch.load(path, map_location="cpu", weights_only=False)
    current = model.state_dict()
    current.update(state["model"])
    model.load_state_dict(current)
    if optimizer is not None:
        optimizer.load_state_dict(state["optimizer"])
    if scheduler is not None:
        scheduler.load_state_dict(state["scheduler"])
    _restore_rng(state["rng"])
    return state["step"], state.get("extra", {})


def initialize_run_dir(
    run_dir: str | Path,
    config: Any,
    manifests: list[str | Path],
    seed: int,
    results_dir: str | Path,
):
    import transformers

    try:
        import peft

        peft_version = peft.__version__
    except ImportError:
        peft_version = None
    run = Path(run_dir)
    results = Path(results_dir)
    run.mkdir(parents=True, exist_ok=True)
    results.mkdir(parents=True, exist_ok=True)
    raw = asdict(config) if is_dataclass(config) else config
    (run / "resolved_config.json").write_text(json.dumps(raw, indent=2, default=str))
    hashes = {
        str(path): hashlib.sha256(Path(path).read_bytes()).hexdigest()
        for path in manifests
        if Path(path).is_file()
    }
    metadata = {
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "peft": peft_version,
        "manifest_hashes": hashes,
        "seed": seed,
    }
    (run / "run_metadata.json").write_text(json.dumps(metadata, indent=2))
    metrics = results / "metrics.json"
    if not metrics.exists():
        metrics.write_text("[]\n")


def _cfg_values(cfg: Any) -> tuple[dict[str, Any], Path]:
    train = cfg.train if hasattr(cfg, "train") else cfg.get("train", cfg)
    results_value = train.get("results_dir", "artifacts/results")
    return train, Path(results_value)


def _delta_state(model: torch.nn.Module) -> dict[str, Any]:
    return {
        name: module.delta_state_dict()
        for name, module in model.named_modules()
        if hasattr(module, "delta_state_dict")
    }


def _new_module_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    excluded = ("language_model.", "encoder.")
    state = {
        name: value.detach().cpu()
        for name, value in model.state_dict().items()
        if not name.startswith(excluded) and ".delta" not in name and "lora_" not in name
    }
    # Fine-tuned encoder blocks are saved; the frozen remainder reloads from the MedSAM weights.
    for name, parameter in model.named_parameters():
        if name.startswith("encoder.") and parameter.requires_grad:
            state[name] = parameter.detach().cpu()
    return state


def save_training_checkpoint(
    directory: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    step: int,
    epoch: int,
    skipped_nonfinite: int,
    epoch_batches: int | None = None,
) -> None:
    model = _unwrap(model)
    destination = Path(directory)
    destination.mkdir(parents=True, exist_ok=True)
    adapter = destination / "lora_adapter"
    adapter.mkdir(exist_ok=True)
    language_model = getattr(model, "language_model", None)
    if language_model is not None and hasattr(language_model, "peft_config"):
        language_model.save_pretrained(adapter)
    else:
        lora = {
            name: value.detach().cpu()
            for name, value in model.state_dict().items()
            if "lora_" in name
        }
        torch.save(lora, adapter / "adapter_model.pt")
    torch.save(_delta_state(model), destination / "vocab_deltas.pt")
    torch.save(_new_module_state(model), destination / "new_modules.pt")
    torch.save(
        {
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "rng": _rng_state(),
        },
        destination / "optimizer.pt",
    )
    trainer_state = {
        "step": step,
        "epoch": epoch,
        "skipped_nonfinite": skipped_nonfinite,
    }
    if epoch_batches is not None:
        trainer_state["epoch_batches"] = epoch_batches  # per-rank batches consumed this epoch
    (destination / "trainer_state.json").write_text(json.dumps(trainer_state, indent=2))


_PEFT_PREFIX = "language_model.base_model.model."


def _peft_aliases(name: str) -> list[str]:
    """Same module before/after PEFT wrapping (stage A has no LoRA, stage B does)."""
    names = [name]
    if name.startswith(_PEFT_PREFIX):
        names.append("language_model." + name[len(_PEFT_PREFIX):])
    elif name.startswith("language_model."):
        names.append(_PEFT_PREFIX + name[len("language_model."):])
    return names


def _load_new_modules(model: torch.nn.Module, path: Path) -> None:
    state = torch.load(path, map_location="cpu", weights_only=False)
    own = model.state_dict()
    remapped = {}
    for key, value in state.items():
        match = next((alias for alias in _peft_aliases(key) if alias in own), None)
        if match is None:
            raise KeyError(f"checkpoint key {key!r} matches no model parameter")
        remapped[match] = value
    model.load_state_dict(remapped, strict=False)


def _load_vocab_deltas(model: torch.nn.Module, path: Path) -> None:
    delta_state = torch.load(path, map_location="cpu", weights_only=False)
    modules = dict(model.named_modules())
    for name, state in delta_state.items():
        match = next((alias for alias in _peft_aliases(name) if alias in modules), None)
        if match is None:
            raise KeyError(f"vocab delta module {name!r} not found in model")
        modules[match].load_delta_state_dict(state)


def load_training_checkpoint(
    directory: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
) -> dict[str, Any]:
    model = _unwrap(model)
    source = Path(directory)
    _load_new_modules(model, source / "new_modules.pt")
    _load_vocab_deltas(model, source / "vocab_deltas.pt")
    adapter_file = source / "lora_adapter" / "adapter_model.pt"
    if adapter_file.exists():
        lora_state = torch.load(adapter_file, map_location="cpu", weights_only=False)
        model.load_state_dict(lora_state, strict=False)
    adapter_safetensors = source / "lora_adapter" / "adapter_model.safetensors"
    if adapter_safetensors.exists():
        from peft import set_peft_model_state_dict
        from safetensors.torch import load_file

        language_model = getattr(model, "language_model", None)
        if language_model is None or not hasattr(language_model, "peft_config"):
            raise ValueError("checkpoint contains LoRA weights but model has no PEFT adapter")
        set_peft_model_state_dict(
            language_model,
            load_file(str(adapter_safetensors), device="cpu"),
            adapter_name="default",
        )
    state = torch.load(source / "optimizer.pt", map_location="cpu", weights_only=False)
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    _restore_rng(state["rng"])
    return json.loads((source / "trainer_state.json").read_text())


def load_component_checkpoint(directory: str | Path, model: torch.nn.Module) -> None:
    """Initialize trainable components without restoring optimizer/trainer state."""
    source = Path(directory)
    model = _unwrap(model)
    _load_new_modules(model, source / "new_modules.pt")
    _load_vocab_deltas(model, source / "vocab_deltas.pt")
    adapter_file = source / "lora_adapter" / "adapter_model.pt"
    if adapter_file.exists():
        model.load_state_dict(
            torch.load(adapter_file, map_location="cpu", weights_only=False),
            strict=False,
        )
    adapter_safetensors = source / "lora_adapter" / "adapter_model.safetensors"
    if adapter_safetensors.exists():
        from peft import set_peft_model_state_dict
        from safetensors.torch import load_file

        language_model = getattr(model, "language_model", None)
        if language_model is None or not hasattr(language_model, "peft_config"):
            raise ValueError("checkpoint contains LoRA weights but model has no PEFT adapter")
        set_peft_model_state_dict(
            language_model,
            load_file(str(adapter_safetensors), device="cpu"),
            adapter_name="default",
        )


def _append_metric(path: Path, record: dict[str, Any]) -> None:
    if path.exists():
        current = json.loads(path.read_text())
        if not isinstance(current, list):
            current = [current]
    else:
        current = []
    current.append(record)
    path.write_text(json.dumps(current, indent=2) + "\n")


def _dice(logits: torch.Tensor, target: torch.Tensor) -> float:
    predicted = torch.sigmoid(logits[:, 0]) > 0.5
    truth = target.to(predicted.device).bool()
    intersection = (predicted & truth).sum().float()
    denominator = predicted.sum() + truth.sum()
    return float((2 * intersection + 1) / (denominator + 1))


@torch.no_grad()
def validate(model, loader, device: torch.device) -> dict[str, float]:
    model.eval()
    sums = torch.zeros(6, device=device, dtype=torch.float64)
    for batch in loader:
        batch = _move_batch(batch, device)
        output = model(batch)
        if output.logits.shape[-1] > 1:
            sums[0] += language_loss(output.logits.float(), output.labels).double()
            sums[1] += 1
        for action in output.actions:
            if action.get("target") is None:
                continue
            kind = "2d" if action["target"].shape[0] == 1 else "3d"
            offset = 2 if kind == "2d" else 4
            sums[offset] += _dice(action["logits"], action["target"])
            sums[offset + 1] += 1
    if _distributed():
        dist.all_reduce(sums, op=dist.ReduceOp.SUM)
    model.train()
    return {
        "ce": float(sums[0] / sums[1].clamp_min(1)),
        "dice_2d": float(sums[2] / sums[3].clamp_min(1)),
        "dice_3d": float(sums[4] / sums[5].clamp_min(1)),
    }


def _move_batch(value: Any, device: torch.device):
    if torch.is_tensor(value):
        return value.to(device, non_blocking=True)
    if isinstance(value, dict):
        return {key: _move_batch(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [_move_batch(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(_move_batch(item, device) for item in value)
    return value


def fit(model, train_loader, val_loader, cfg):
    """Train for optimizer steps and resume from the component checkpoint if requested."""
    train_cfg, default_results = _cfg_values(cfg)
    results_dir = Path(train_cfg.get("results_dir", default_results))
    rank = _rank()
    if rank == 0:
        results_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = results_dir / "metrics.json"
    device = next(model.parameters()).device
    max_steps = int(train_cfg.get("max_steps", 1))
    grad_accum = int(train_cfg.get("gradient_accumulation", 1))
    groups, counts = parameter_groups(
        model,
        float(train_cfg.get("learning_rate", 1e-4)),
        float(train_cfg.get("encoder_learning_rate", 0.0)),
        train_cfg.get("lora_learning_rate"),
        train_cfg.get("delta_learning_rate"),
    )
    if not groups:
        raise ValueError("model has no trainable parameter groups")
    optimizer = torch.optim.AdamW(
        groups,
        weight_decay=float(train_cfg.get("weight_decay", 0.0)),
    )
    scheduler = cosine_scheduler(
        optimizer,
        int(train_cfg.get("warmup_steps", 0)),
        max_steps,
    )
    state = {"step": 0, "epoch": 0, "skipped_nonfinite": 0, "epoch_batches": 0}
    resume = train_cfg.get("resume")
    resume_skip = 0
    if resume:
        state.update(load_training_checkpoint(resume, model, optimizer, scheduler))
        if "epoch_batches" not in state or state["epoch_batches"] is None:
            # Older checkpoints: infer from optimizer steps (exact unless non-finite skips occurred).
            accumulation = int(train_cfg.get("gradient_accumulation", 1))
            state["epoch_batches"] = max(
                0, state["step"] * accumulation - state["epoch"] * len(train_loader))
        resume_skip = int(state["epoch_batches"])
    elif train_cfg.get("init_from"):
        load_component_checkpoint(train_cfg["init_from"], model)

    if rank == 0:
        (results_dir / "trainable_parameters.json").write_text(
            json.dumps(counts, indent=2) + "\n"
        )
    if _distributed():
        dist.barrier()
    checkpoint_root_value = train_cfg.get("checkpoint_dir")
    checkpoint_root = Path(checkpoint_root_value or results_dir)
    checkpoint_every = int(train_cfg.get("checkpoint_every", max_steps))
    validate_every = int(train_cfg.get("validate_every", max_steps))
    clip_norm = float(train_cfg.get("gradient_clip_norm", 1.0))
    text_weight = float(train_cfg.get("loss", {}).get("text", 1.0))
    mask_weight = float(train_cfg.get("loss", {}).get("mask", 1.0))
    empty_policy = train_cfg.get("empty_target_policy", "include")
    negative_weight = float((train_cfg.get("loss") or {}).get("negative", 0.0))
    optimizer.zero_grad(set_to_none=True)
    micro_step = 0
    last_loss = 0.0
    step_started = time.perf_counter()

    while state["step"] < max_steps:
        made_progress = False
        sampler = getattr(train_loader, "batch_sampler", None)
        if hasattr(sampler, "set_epoch"):
            sampler.set_epoch(state["epoch"])
        if resume_skip and hasattr(sampler, "start_batch"):
            # Continue the interrupted epoch instead of replaying its first batches.
            sampler.start_batch = resume_skip
        elif resume_skip:
            print("[train] sampler cannot skip; resumed epoch replays its first batches",
                  file=sys.stderr, flush=True)
            state["epoch_batches"], resume_skip = 0, 0
        if not resume_skip:
            state["epoch_batches"] = 0
        resumed_mid_epoch, resume_skip = bool(resume_skip), 0
        if hasattr(train_loader.dataset, "config"):
            train_loader.dataset.config.epoch = state["epoch"]
        for raw_batch in train_loader:
            made_progress = True
            state["epoch_batches"] += 1
            batch = _move_batch(raw_batch, device)
            use_autocast = device.type == "cuda"
            autocast = (
                torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                if use_autocast
                else nullcontext()
            )
            final_micro_step = (micro_step + 1) % grad_accum == 0
            sync_context = (
                model.no_sync()
                if hasattr(model, "no_sync") and not final_micro_step
                else nullcontext()
            )
            debug = os.environ.get("PRSMED15_DEBUG_STEPS") == "1"
            if debug:
                items = batch.get("items", [])
                print(json.dumps({
                    "debug": "micro_start", "rank": rank, "micro": micro_step, "t": round(time.time(), 1),
                    "ids": [item.get("dialogue_id") for item in items],
                    "seq_len": int(batch["input_ids"].shape[1]),
                    "depth": int(batch["pixel_values"].shape[2]),
                    "turns": [len(item.get("turn_boundaries", [])) for item in items],
                    "actions": len(batch.get("actions", [])),
                }), file=sys.stderr, flush=True)
            with sync_context:
                with autocast:
                    try:
                        output = model(batch)
                    except Exception:
                        ids = [item.get("dialogue_id") for item in batch.get("items", [])]
                        print(f"[rank {rank}] forward failed on dialogues {ids}", file=sys.stderr, flush=True)
                        raise
                    loss, pieces = total_loss(
                        output.logits.float(),
                        output.labels,
                        output.actions,
                        lambda_text=text_weight,
                        lambda_mask=mask_weight,
                        empty_policy=empty_policy,
                        lambda_negative=negative_weight,
                    )
                finite_loss = _collective_all(bool(torch.isfinite(loss)), device)
                if finite_loss:
                    (loss / grad_accum).backward()
            if not finite_loss:
                optimizer.zero_grad(set_to_none=True)
                state["skipped_nonfinite"] += 1
                micro_step = 0
                continue
            micro_step += 1
            last_loss = float(loss.detach())
            if debug:
                print(json.dumps({"debug": "micro_end", "rank": rank, "t": round(time.time(), 1)}),
                      file=sys.stderr, flush=True)
            if micro_step % grad_accum:
                continue
            average_gradients(model)
            grads = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
            # One device->host sync instead of one per parameter.
            finite_gradients = (not grads) or bool(
                torch.stack([torch.isfinite(grad).all() for grad in grads]).all())
            finite_gradients = _collective_all(finite_gradients, device)
            if not finite_gradients:
                optimizer.zero_grad(set_to_none=True)
                state["skipped_nonfinite"] += 1
                micro_step = 0
                continue
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            state["step"] += 1
            record = {
                "split": "train",
                "step": state["step"],
                "loss": last_loss,
                "ce": float(pieces["ce"].detach()),
                "mask": float(pieces["mask"].detach()),
                **output.history_fractions,
                "step_seconds": round(time.perf_counter() - step_started, 2),
                "seq_len": int(batch["input_ids"].shape[1]),
                "depth": int(batch["pixel_values"].shape[2]),
            }
            step_started = time.perf_counter()
            with torch.no_grad():
                dices = [_dice(a["logits"].float(), a["target"]) for a in output.actions
                         if a.get("target") is not None]
            record["train_dice"] = float(sum(dices) / len(dices)) if dices else None
            record["n_actions"] = len(dices)
            if rank == 0:
                print(json.dumps(record), flush=True)
                _append_metric(metrics_path, record)
            if val_loader is not None and state["step"] % validate_every == 0:
                validation = validate(model, val_loader, device)
                if rank == 0:
                    _append_metric(
                        metrics_path,
                        {"split": "val", "step": state["step"], **validation},
                    )
            should_checkpoint = (
                state["step"] % checkpoint_every == 0 or state["step"] == max_steps
            )
            if rank == 0 and should_checkpoint:
                checkpoint_path = (
                    checkpoint_root / f"checkpoint-{state['step']}"
                    if checkpoint_root_value
                    else checkpoint_root / "checkpoint"
                )
                save_training_checkpoint(
                    checkpoint_path,
                    model,
                    optimizer,
                    scheduler,
                    state["step"],
                    state["epoch"],
                    state["skipped_nonfinite"],
                    state["epoch_batches"],
                )
            if _distributed() and should_checkpoint:
                dist.barrier()
            if state["step"] >= max_steps:
                break
        if not made_progress and not resumed_mid_epoch:
            raise ValueError("train_loader produced no batches")
        state["epoch"] += 1
    return {
        "step": state["step"],
        "loss": last_loss,
        "skipped_nonfinite": state["skipped_nonfinite"],
        "trainable_parameters": counts,
    }


class Trainer:
    """Compatibility wrapper used by the original Phase 2 unit tests."""

    def __init__(self, model, optimizer, scheduler, grad_accum: int = 1):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.grad_accum = grad_accum
        self.step = 0
        self.skipped_nonfinite = 0

    def backward(self, loss):
        loss.div(self.grad_accum).backward()

    def optimizer_step(self):
        finite = all(
            parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
            for parameter in self.model.parameters()
        )
        if not finite:
            self.optimizer.zero_grad(set_to_none=True)
            self.skipped_nonfinite += 1
            return False
        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad(set_to_none=True)
        self.step += 1
        return True
