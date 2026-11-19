"""Padding collator for a single input-kind bucket."""
from __future__ import annotations

from typing import Any

import torch


def collate_dialogues(items: list[dict[str, Any]], pad_token_id: int = 0) -> dict[str, Any]:
    if not items:
        raise ValueError("cannot collate an empty batch")
    kinds = {item["input_kind"] for item in items}
    if len(kinds) != 1:
        raise ValueError(f"batch must be bucketed by input_kind, got {sorted(kinds)}")
    if any(len(item["scans"]) != 1 for item in items):
        raise ValueError("Phase 1 collator supports one scan per dialogue")
    b = len(items)
    max_c = max(x["scans"][0].shape[0] for x in items)
    max_d = max(x["scans"][0].shape[1] for x in items)
    max_h = max(x["scans"][0].shape[2] for x in items)
    max_w = max(x["scans"][0].shape[3] for x in items)
    pixels = torch.zeros((b, max_c, max_d, max_h, max_w), dtype=torch.float32)
    valid = torch.zeros((b, max_d, max_h, max_w), dtype=torch.bool)
    for i, item in enumerate(items):
        x, mask = item["scans"][0], item["valid_masks"][0]
        c, d, h, w = x.shape
        pixels[i, :c, :d, :h, :w] = x
        valid[i, :d, :h, :w] = mask
    max_len = max(x["input_ids"].numel() for x in items)
    input_ids = torch.full((b, max_len), pad_token_id, dtype=torch.long)
    labels = torch.full((b, max_len), -100, dtype=torch.long)
    attention = torch.zeros((b, max_len), dtype=torch.bool)
    actions: list[dict[str, Any]] = []
    memory_records: list[dict[str, Any]] = []
    for i, item in enumerate(items):
        n = item["input_ids"].numel()
        input_ids[i, :n] = item["input_ids"]
        labels[i, :n] = item["labels"]
        attention[i, :n] = True
        actions.extend({**action, "batch_idx": i} for action in item["actions"])
        memory_records.extend(
            {**record, "batch_idx": i}
            for record in item.get("memory_records", [])
        )
    # Cached encoder features travel once at top level (not again inside "items").
    cached = [item.pop("encoder_features", None) for item in items]
    features = None
    if all(value is not None for value in cached):
        h, w, c = cached[0][0].shape[1:]
        features = torch.zeros((b, max_d, h, w, c), dtype=cached[0][0].dtype)
        for i, value in enumerate(cached):
            features[i, :value[0].shape[0]] = value[0]
    return {"input_kind": kinds.pop(), "encoder_features": features, "pixel_values": pixels, "pixel_valid_mask": valid,
            "input_ids": input_ids, "labels": labels, "attention_mask": attention,
            "actions": actions, "memory_records": memory_records, "items": items}
