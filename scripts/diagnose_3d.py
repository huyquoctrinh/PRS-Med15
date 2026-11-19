"""Diagnose 3D mask quality of a training checkpoint (per action, per slice, alignment overlays).

Usage: CUDA_VISIBLE_DEVICES=3 PRSMED15_ALLOW_HEAVY=1 PYTHONPATH=src python scripts/diagnose_3d.py \
    CKPT OUT_DIR [n_val_per_source] [n_train_per_source]
"""
from __future__ import annotations

from functools import partial
import json
from pathlib import Path
import random
import sys

import numpy as np
import torch

from prs_med15.cli import PROJECT_ROOT, _dataset_config, _read_manifest
from prs_med15.config import load_config
from prs_med15.data.collate import collate_dialogues
from prs_med15.data.dataset import DialogueDataset
from prs_med15.models.factory import build_model
from prs_med15.training.trainer import _move_batch, load_component_checkpoint


def _dice(p: torch.Tensor, t: torch.Tensor) -> float:
    inter = (p & t).sum().item()
    den = p.sum().item() + t.sum().item()
    return 2 * inter / den if den else float("nan")


def stats(logits: torch.Tensor, target: torch.Tensor, valid: torch.Tensor) -> dict:
    prob = torch.sigmoid(logits[:, 0].float())
    pred = prob > 0.5
    t = target.to(pred.device).bool()
    tslices = t.flatten(1).any(1)
    pslices = pred.flatten(1).any(1)
    per_slice = [_dice(pred[i], t[i]) for i in range(t.shape[0]) if tslices[i]]
    on = tslices
    return {
        "depth": int(t.shape[0]),
        "target_vox": int(t.sum()),
        "target_slices": int(tslices.sum()),
        "pred_vox": int(pred.sum()),
        "pred_slices": int(pslices.sum()),
        "dice_vol": _dice(pred, t),
        "dice_on_target_slices": _dice(pred[on], t[on]),
        "mean_slice_dice": float(np.mean(per_slice)) if per_slice else float("nan"),
        "fp_vox_off_slices": int(pred[~on].sum()),
        "fp_slices_off": int((pslices & ~tslices).sum()),
        "max_prob_off": float(prob[~on].max()) if (~on).any() else float("nan"),
        "max_prob_on": float(prob[on].max()),
        "pred_vox_outside_valid": int((pred & ~valid.to(pred.device).bool()).sum()),
        # Oracle-threshold upper bound: best Dice achievable by thresholding this probability map.
        "best_thr_dice": max(_dice(prob > th, t) for th in (0.1, 0.3, 0.5, 0.7, 0.9)),
    }


def overlay(path: Path, image: torch.Tensor, target: torch.Tensor, logits: torch.Tensor, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t = target.bool().cpu()
    z = int(t.flatten(1).sum(1).argmax())
    img = image[0, z].float().cpu().numpy() if image.ndim == 4 else image[z].float().cpu().numpy()
    pred = (torch.sigmoid(logits[z, 0].float()) > 0.5).cpu().numpy()
    fig, ax = plt.subplots(1, 3, figsize=(12, 4.3))
    ax[0].imshow(img, cmap="gray"); ax[0].set_title(f"image z={z}")
    ax[1].imshow(img, cmap="gray"); ax[1].contour(t[z].numpy(), colors="lime", linewidths=0.8)
    ax[1].set_title("target (green)")
    ax[2].imshow(img, cmap="gray"); ax[2].contour(pred, colors="red", linewidths=0.8)
    ax[2].contour(t[z].numpy(), colors="lime", linewidths=0.5); ax[2].set_title("pred (red)")
    for a in ax: a.axis("off")
    fig.suptitle(title, fontsize=9); fig.tight_layout(); fig.savefig(path, dpi=80); plt.close(fig)


def main() -> None:
    ckpt, out = sys.argv[1], Path(sys.argv[2])
    n_val = int(sys.argv[3]) if len(sys.argv) > 3 else 30
    n_train = int(sys.argv[4]) if len(sys.argv) > 4 else 15
    out.mkdir(parents=True, exist_ok=True)
    cfg = load_config("configs/train/full_stage_b.yaml", [])
    cfg.model = load_config(PROJECT_ROOT / "configs/model/shared_slice.yaml").model
    model, tokenizer, _ = build_model(cfg, tiny=False)
    load_component_checkpoint(ckpt, model)
    device = torch.device("cuda")
    model = model.to(device).eval()
    alpha = float(model.adapter.alpha) if hasattr(model.adapter, "alpha") else None
    print(json.dumps({"adapter_alpha": alpha}), flush=True)
    rng = random.Random(0)
    collate = partial(collate_dialogues, pad_token_id=tokenizer.pad_token_id or 0)
    dcfg = _dataset_config(cfg, model, False)
    rows_out = []
    n_overlay = 0
    for split, n, mode in (("val", n_val, "eval"), ("train", n_train, "train")):
        for source in ("totalseg", "totalseg_mr", "kits", "amos"):
            path = Path(f"data/manifests/{source}_{split}.jsonl")
            if not path.exists() or path.stat().st_size == 0:
                continue
            rows = _read_manifest(path)
            rows = rng.sample(rows, min(n, len(rows)))
            dataset = DialogueDataset(rows, tokenizer, dcfg, mode=mode)
            for index in range(len(dataset)):
                try:
                    item = dataset[index]
                except Exception as error:  # noqa: BLE001
                    print(json.dumps({"skip": rows[index].dialogue_id, "error": str(error)[:200]}), flush=True)
                    continue
                batch = _move_batch(collate([item]), device)
                with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    output = model(batch)
                dialogue = rows[index]
                geo = item["geometry"][0]
                kept = [(a["entity_id"], torch.sigmoid(a["logits"][:, 0].float()) > 0.5, a["target"].to(device).bool())
                        for a in output.actions if a.get("target") is not None and a.get("dimension") == "3d"]
                for k, action in enumerate(output.actions):
                    target = action.get("target")
                    if target is None or action.get("dimension") != "3d":
                        continue
                    obj = dialogue.objects.get(action["entity_id"])
                    record = {
                        "split": split, "source": source, "dialogue_id": dialogue.dialogue_id,
                        "entity": obj.canonical if obj else action["entity_id"],
                        "turn_index": action["turn_index"], "op": action["op"],
                        "n_refs": len(action.get("references") or []),
                        "original_depth": geo.get("original_depth"), "slab_start": geo.get("slab_start"),
                        **stats(action["logits"], target, action["valid"]),
                    }
                    pred = torch.sigmoid(action["logits"][:, 0].float()) > 0.5
                    t = target.to(device).bool()
                    others = [(e, _dice(pred, ot)) for e, _, ot in kept if e != action["entity_id"]]
                    if others:
                        e, d = max(others, key=lambda x: x[1])
                        record["best_other_dice"], record["best_other_entity"] = d, dialogue.objects[e].canonical
                    record["dice_lr_flip"] = _dice(pred.flip(-1), t)
                    record["dice_ap_flip"] = _dice(pred.flip(-2), t)
                    rows_out.append(record)
                    print(json.dumps(record), flush=True)
                    if n_overlay < 24 and k == 0:
                        overlay(out / f"{split}_{source}_{n_overlay:02d}_{record['entity']}.png",
                                item["scans"][0], target, action["logits"],
                                f"{split} {source} {record['entity']} dice={record['dice_vol']:.2f} "
                                f"on-slices={record['dice_on_target_slices']:.2f}")
                        n_overlay += 1
    (out / "actions.json").write_text(json.dumps(rows_out, indent=1))


if __name__ == "__main__":
    main()
