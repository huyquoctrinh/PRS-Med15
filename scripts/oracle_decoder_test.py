"""Capacity test without the LLM: fresh slice adapter + MedSAM decoder conditioned on an oracle
per-class embedding, trained and scored on the same few 3D dialogues. Variants change the input
the frozen encoder sees.

Usage: CUDA_VISIBLE_DEVICES=3 PRSMED15_ALLOW_HEAVY=1 PYTHONPATH=src python scripts/oracle_decoder_test.py \
    MANIFEST VARIANT[,VARIANT...] [iters]
Variants: wide (current [-1000,1000] window), soft ([-160,240]), wide_rot / soft_rot (180° in-plane);
suffix _enc4 trains MedSAM encoder blocks 9-12 + neck (lr ENC_LR, default 3e-5) with online encoding.
"""
from __future__ import annotations

import dataclasses
import json
import random
import sys

import torch
from torch import nn

from prs_med15.cli import PROJECT_ROOT, _dataset_config, _read_manifest
from prs_med15.config import load_config
from prs_med15.data.dataset import DialogueDataset
from prs_med15.models.factory import build_model
from prs_med15.training.losses import action_mask_loss


def window(x: torch.Tensor, variant: str) -> torch.Tensor:
    if variant.startswith("soft"):
        hu = x * 2000.0 - 1000.0
        x = ((hu + 160.0) / 400.0).clamp(0, 1)
    return x


def main() -> None:
    manifest, variants = sys.argv[1], sys.argv[2].split(",")
    iters = int(sys.argv[3]) if len(sys.argv) > 3 else 600
    cfg = load_config("configs/train/full_stage_b2.yaml", [])
    torch.manual_seed(0)
    model, tokenizer, _ = build_model(cfg, tiny=False)
    device = torch.device("cuda")
    rows = _read_manifest(manifest)
    dcfg = dataclasses.replace(_dataset_config(cfg, model, False), feature_root=None)
    dataset = DialogueDataset(rows, tokenizer, dcfg, mode="train")
    items = [dataset[i] for i in range(len(dataset))]
    encoder = model.encoder.to(device).eval()
    init_adapter = {k: v.clone() for k, v in model.adapter.state_dict().items()}
    init_decoder = {k: v.clone() for k, v in model.mask_decoder.state_dict().items()}
    import os
    init_encoder = {k: v.clone() for k, v in encoder.state_dict().items()}
    for variant in variants:
        train_encoder = variant.endswith("_enc4")
        base_variant = variant.removesuffix("_enc4")
        rot = base_variant.endswith("_rot")
        encoder.load_state_dict(init_encoder)
        encoder.requires_grad_(False)
        trainable_encoder = []
        if train_encoder:
            for module in list(encoder.image_encoder.layers[8:]) + [encoder.image_encoder.neck]:
                module.requires_grad_(True)
                trainable_encoder += list(module.parameters())
        samples, classes = [], {}
        for item, row in zip(items, rows):
            image = window(item["scans"][0], base_variant)
            valid = item["valid_masks"][0]
            if rot:
                image, valid = image.flip(-1, -2), valid.flip(-1, -2)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                bundle = encoder(image[None].to(device), valid[None].to(device))
            geometry = [item["geometry"][0]]
            for action in item["actions"]:
                target = action["target"]
                if target is None or target.shape[0] == 1:
                    continue
                name = row.objects[action["entity_id"]].canonical
                classes.setdefault(name, len(classes))
                samples.append({
                    "dense": None if train_encoder else bundle.dense.float(), "valid": bundle.valid_mask,
                    "image": image if train_encoder else None, "pixel_valid": valid.to(device),
                    "geometry": geometry, "target": (target.flip(-1, -2) if rot else target).to(device),
                    "cls": classes[name],
                })
        model.adapter.load_state_dict(init_adapter)
        model.mask_decoder.load_state_dict(init_decoder)
        adapter, decoder = model.adapter.to(device).train(), model.mask_decoder.to(device).train()
        emb = nn.Embedding(len(classes), model.language_model.config.hidden_size).to(device)
        nn.init.normal_(emb.weight, std=1.0)
        optim = torch.optim.AdamW([
            {"params": list(adapter.parameters()) + list(decoder.parameters()), "lr": 1e-4},
            {"params": emb.parameters(), "lr": 1e-3},
        ] + ([{"params": trainable_encoder, "lr": float(os.environ.get("ENC_LR", "3e-5"))}] if trainable_encoder else []),
            weight_decay=0.0)
        rng = random.Random(0)

        def run(sample):
            features = sample["dense"]
            if features is None:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    features = encoder(sample["image"][None].to(device), sample["pixel_valid"][None]).dense.float()
            pos = model._depth_positions(sample["geometry"], features.shape[1], device)
            dense = adapter(features, sample["valid"], pos)
            state = emb.weight[sample["cls"]]
            return decoder(dense[0].permute(0, 3, 1, 2), state, (512, 512), sample["pixel_valid"])

        def score() -> float:
            adapter.eval(); decoder.eval()
            values = []
            with torch.no_grad():
                for sample in samples:
                    pred = torch.sigmoid(run(sample)[:, 0]) > 0.5
                    t = sample["target"].bool()
                    values.append((2 * (pred & t).sum() / (pred.sum() + t.sum()).clamp_min(1)).item())
            adapter.train(); decoder.train()
            return sum(values) / len(values)

        log = {"variant": variant, "n_samples": len(samples), "n_classes": len(classes), "dice": {}}
        for step in range(1, iters + 1):
            sample = rng.choice(samples)
            loss = action_mask_loss(run(sample), sample["target"], sample["pixel_valid"])
            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()
            if step % (iters // 4) == 0:
                log["dice"][step] = round(score(), 3)
                print(json.dumps({"variant": variant, "step": step, "dice": log["dice"][step]}), flush=True)
        print(json.dumps(log), flush=True)


if __name__ == "__main__":
    main()
