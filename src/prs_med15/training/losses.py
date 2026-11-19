"""Dialogue-aware language and valid-voxel mask losses.

Dice/BCE formulas are adapted with attribution from PRS-Med trainers/metrics.py.
"""
from __future__ import annotations
from collections import defaultdict
import torch
import torch.nn.functional as F


def language_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits[:,:-1].reshape(-1,logits.shape[-1]),labels[:,1:].reshape(-1),ignore_index=-100)


def action_mask_loss(logits: torch.Tensor, target: torch.Tensor, valid: torch.Tensor | None = None,
                     empty_policy: str = "include") -> torch.Tensor | None:
    target=target.float(); valid=torch.ones_like(target,dtype=torch.bool) if valid is None else valid.bool()
    if empty_policy=="skip" and not bool((target*valid).any()): return None
    if empty_policy not in {"skip","include"}: raise ValueError("empty_policy must be skip or include")
    pred=logits.squeeze(1) if logits.ndim==target.ndim+1 else logits
    bce=F.binary_cross_entropy_with_logits(pred[valid],target[valid])
    prob=torch.sigmoid(pred)*valid; truth=target*valid
    dice=1-(2*(prob*truth).sum()+1)/(prob.sum()+truth.sum()+1)
    return bce+dice


def hard_negative_loss(actions: list[dict]) -> list[torch.Tensor | None]:
    """Per action: mean foreground probability on voxels of OTHER entities of the same scan that are
    not part of its own target (forces the [SEG] query to select the asked entity)."""
    out: list[torch.Tensor | None] = []
    for i, action in enumerate(actions):
        own = action.get("target")
        if own is None:
            out.append(None); continue
        key = (action.get("batch_idx", 0), action.get("scan_idx", 0))
        negative = None
        for j, other in enumerate(actions):
            target = other.get("target")
            if j == i or target is None or other.get("entity_id") == action.get("entity_id"):
                continue
            if (other.get("batch_idx", 0), other.get("scan_idx", 0)) != key or target.shape != own.shape:
                continue
            negative = target.bool() if negative is None else negative | target.bool()
        if negative is None:
            out.append(None); continue
        negative = negative & ~own.bool()
        if action.get("valid") is not None:
            negative = negative & action["valid"].bool()
        if not bool(negative.any()):
            out.append(None); continue
        logits = action["logits"]
        pred = logits.squeeze(1) if logits.ndim == own.ndim + 1 else logits
        out.append(torch.sigmoid(pred.float())[negative].mean())
    return out


def dialogue_mask_loss(actions: list[dict], reference: torch.Tensor, empty_policy: str = "include",
                       lambda_negative: float = 0.0) -> torch.Tensor:
    grouped=defaultdict(list)
    negatives = hard_negative_loss(actions) if lambda_negative > 0 else [None] * len(actions)
    for action, negative in zip(actions, negatives):
        if action.get("target") is None: continue
        value=action_mask_loss(action["logits"],action["target"],action.get("valid"),empty_policy)
        if value is not None and negative is not None:
            value = value + lambda_negative * negative
        if value is not None: grouped[action.get("dialogue_id",action.get("batch_idx",0))].append(value)
    if not grouped: return reference.sum()*0.0
    return torch.stack([torch.stack(values).mean() for values in grouped.values()]).mean()


def total_loss(logits,labels,actions,lambda_text=1.,lambda_mask=1.,empty_policy="include",lambda_negative=0.):
    if logits.shape[-1] == 1:
        ce = logits.sum() * 0.0  # text logits not computed (text weight 0)
    else:
        ce = language_loss(logits, labels)
    mask=dialogue_mask_loss(actions,logits,empty_policy,lambda_negative)
    return lambda_text*ce+lambda_mask*mask,{"ce":ce,"mask":mask}
