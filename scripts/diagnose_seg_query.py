"""Does the [SEG] query carry entity identity? Cosine of seg states / decoder queries across entities
of one dialogue, and how much the decoder output changes when the query is swapped.

Usage: CUDA_VISIBLE_DEVICES=3 PRSMED15_ALLOW_HEAVY=1 PYTHONPATH=src python scripts/diagnose_seg_query.py CKPT [n_per_source]
"""
from __future__ import annotations

from functools import partial
import itertools
import json
import random
import sys

import torch
import torch.nn.functional as F

from prs_med15.cli import PROJECT_ROOT, _dataset_config, _read_manifest
from prs_med15.config import load_config
from prs_med15.data.collate import collate_dialogues
from prs_med15.data.dataset import DialogueDataset
from prs_med15.models.factory import build_model
from prs_med15.training.trainer import _move_batch, load_component_checkpoint


def main() -> None:
    ckpt = sys.argv[1]
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    import os
    cfg = load_config(os.environ.get("DIAG_CONFIG", "configs/train/full_stage_b.yaml"), [])
    if not cfg.model:
        cfg.model = load_config(PROJECT_ROOT / "configs/model/shared_slice.yaml").model
    model, tokenizer, _ = build_model(cfg, tiny=False)
    load_component_checkpoint(ckpt, model)
    device = torch.device("cuda")
    model = model.to(device).eval()
    calls = []

    def hook(module, args, kwargs, out):
        features, seg_state = args[0], args[1]
        calls.append((features.detach(), seg_state.detach().float(), out.detach().float(), args[2:], kwargs))

    model.mask_decoder.register_forward_hook(hook, with_kwargs=True)
    collate = partial(collate_dialogues, pad_token_id=tokenizer.pad_token_id or 0)
    dcfg = _dataset_config(cfg, model, False)
    rng = random.Random(1)
    agg = {"cos_hidden": [], "cos_query": [], "out_change_swap": [], "out_change_noise": [], "dice_swap_vs_orig": []}
    dice_own, dice_other = [], []
    manifest = os.environ.get("DIAG_MANIFEST")
    sources = ("subset",) if manifest else ("kits", "totalseg", "totalseg_mr", "mrmedseg")
    for source in sources:
        rows = _read_manifest(manifest or f"data/manifests/{source}_val.jsonl")
        rows = rng.sample(rows, min(n, len(rows)))
        dataset = DialogueDataset(rows, tokenizer, dcfg, mode="eval")
        for index in range(len(dataset)):
            calls.clear()
            batch = _move_batch(collate([dataset[index]]), device)
            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                output = model(batch)
            ents = [a["entity_id"] for a in output.actions]
            tg = [(a["entity_id"], a.get("target")) for a in output.actions]
            for i, (e, t) in enumerate(tg):
                if t is None or i >= len(calls):
                    continue
                p = calls[i][2][:, 0] > 0
                t = t.to(p.device).bool()
                d = lambda x, y: 2 * (x & y).sum().item() / max(x.sum().item() + y.sum().item(), 1)
                dice_own.append(d(p, t))
                others = [d(p, ot.to(p.device).bool()) for oe, ot in tg if oe != e and ot is not None]
                if others:
                    dice_other.append(max(others) > d(p, t))
            if len(calls) < 2:
                continue
            q = [model.mask_decoder.query(c[1].reshape(1, -1)).flatten() for c in calls]
            for i, j in itertools.combinations(range(len(calls)), 2):
                if ents[i] == ents[j]:
                    continue
                agg["cos_hidden"].append(F.cosine_similarity(calls[i][1].flatten(), calls[j][1].flatten(), 0).item())
                agg["cos_query"].append(F.cosine_similarity(q[i], q[j], 0).item())
            # Swap: decode action 0's features with action 1's seg state.
            f0, s0, o0, rest, kw = calls[0]
            s1 = calls[1][1]
            with torch.no_grad():
                swapped = model.mask_decoder(f0, s1, *rest, **kw).float()
                noise = model.mask_decoder(f0, s0 + 0.01 * s0.norm() / s0.numel() ** 0.5 * torch.randn_like(s0), *rest, **kw).float()
            p0, ps = o0 > 0, swapped > 0
            agg["out_change_swap"].append((p0 ^ ps).float().mean().item() / max(p0.float().mean().item(), 1e-6))
            agg["out_change_noise"].append((p0 ^ (noise > 0)).float().mean().item() / max(p0.float().mean().item(), 1e-6))
            inter = (p0 & ps).sum().item(); den = p0.sum().item() + ps.sum().item()
            agg["dice_swap_vs_orig"].append(2 * inter / den if den else 1.0)
            print(source, rows[index].dialogue_id, [rows[index].objects[e].canonical for e in ents][:4],
                  "swap-dice", round(agg["dice_swap_vs_orig"][-1], 3), flush=True)
    agg["dice_own"], agg["frac_other_better"] = dice_own, [float(x) for x in dice_other]
    print(json.dumps({k: (round(sum(v) / len(v), 4), len(v)) for k, v in agg.items() if v}))


if __name__ == "__main__":
    main()
