"""Per-component GPU timing of one training forward/backward on real dialogues.

Usage: CUDA_VISIBLE_DEVICES=3 PRSMED15_ALLOW_HEAVY=1 PYTHONPATH=src python scripts/profile_step.py [n_3d] [n_2d]
"""
from __future__ import annotations

from collections import defaultdict
from functools import partial
import json
import sys
import time

import torch

from prs_med15.cli import PROJECT_ROOT, _dataset_config, _read_manifest
from prs_med15.config import load_config
from prs_med15.data.collate import collate_dialogues
from prs_med15.data.dataset import DialogueDataset
from prs_med15.models.factory import build_model
from prs_med15.training.losses import total_loss
from prs_med15.training.trainer import _move_batch


def main() -> None:
    n3d = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    n2d = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    import os
    cfg = load_config(os.environ.get("PROFILE_CONFIG", "configs/train/full_stage_a.yaml"), [])
    cfg.model = load_config(PROJECT_ROOT / "configs/model/shared_slice.yaml").model
    if "PROFILE_CKPT" in os.environ:  # 0/1 overrides gradient checkpointing
        cfg.train["gradient_checkpointing"] = os.environ["PROFILE_CKPT"] == "1"
    if os.environ.get("PROFILE_FEATURES"):
        cfg.data.feature_root = os.environ["PROFILE_FEATURES"]
    loss_cfg = cfg.train.get("loss", {"text": 0.0, "mask": 1.0})
    model, tokenizer, _ = build_model(cfg, tiny=False)
    device = torch.device("cuda")
    model = model.to(device).train()
    wanted = [x for x in os.environ.get("PROFILE_IDS", "").split(",") if x]
    if wanted:
        rows = []
        for name in ("mrmedseg_train", "totalseg_train", "totalseg_mr_train", "amos_train", "kits_train"):
            rows += [r for r in _read_manifest(f"data/manifests/{name}.jsonl") if r.dialogue_id in wanted]
    else:
        rows = _read_manifest("data/manifests/totalseg_train.jsonl", n3d)
        rows += _read_manifest("data/manifests/mrmedseg_train.jsonl", n2d)
    dataset = DialogueDataset(rows, tokenizer, _dataset_config(cfg, model, False), mode="train")
    collate = partial(collate_dialogues, pad_token_id=tokenizer.pad_token_id or 0)

    timings: dict[str, float] = defaultdict(float)
    calls: dict[str, int] = defaultdict(int)
    starts: dict[str, list[float]] = defaultdict(list)

    def pre(name, *_):
        torch.cuda.synchronize()
        starts[name].append(time.perf_counter())

    def post(name, *_):
        torch.cuda.synchronize()
        timings[name] += time.perf_counter() - starts[name].pop()
        calls[name] += 1

    for name in ("encoder", "adapter", "global_pooler", "query_encoder", "detail_retriever",
                 "language_model", "mask_decoder", "region_pool", "region_tokenizer"):
        module = getattr(model, name)
        module.register_forward_pre_hook(partial(pre, name))
        module.register_forward_hook(partial(post, name))

    for index in range(len(dataset)):
        timings.clear()
        calls.clear()
        t0 = time.perf_counter()
        item = dataset[index]
        t_load = time.perf_counter() - t0
        batch = _move_batch(collate([item]), device)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        t1 = time.perf_counter()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            output = model(batch)
            loss, _ = total_loss(output.logits.float(), output.labels, output.actions,
                                 lambda_text=float(loss_cfg["text"]),
                                 lambda_mask=float(loss_cfg["mask"]))
        torch.cuda.synchronize()
        t2 = time.perf_counter()
        loss.backward()
        torch.cuda.synchronize()
        t3 = time.perf_counter()
        model.zero_grad(set_to_none=True)
        print(json.dumps({
            "source": rows[index].source,
            "seq_len": int(batch["input_ids"].shape[1]),
            "turns": len(batch["items"][0]["turn_boundaries"]),
            "actions": len(batch["actions"]),
            "pixel_shape": list(batch["pixel_values"].shape),
            "load_s": round(t_load, 2),
            "forward_s": round(t2 - t1, 2),
            "backward_s": round(t3 - t2, 2),
            "peak_gb": round(torch.cuda.max_memory_allocated() / 2**30, 1),
            "components_s": {k: round(v, 2) for k, v in timings.items()},
            "calls": dict(calls),
        }), flush=True)


if __name__ == "__main__":
    main()
