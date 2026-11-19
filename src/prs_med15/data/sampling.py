"""Training-only target-aware and leakage-free evaluation slab policies."""
from __future__ import annotations

from typing import Iterable, Sequence
import hashlib
import torch


def centre_slab(depth: int, d_max: int) -> tuple[int, int]:
    start = max(0, (depth - d_max) // 2)
    return start, min(depth, start + d_max)


def training_target_slab(depth: int, d_max: int, target_centres_z: Iterable[float] | None) -> tuple[int, int]:
    """Choose a target-containing slab using GT facts; training only, never evaluation."""
    centres = list(target_centres_z or [])
    if not centres:
        return centre_slab(depth, d_max)
    centre = round(sum(centres) / len(centres))
    start = min(max(0, centre - d_max // 2), max(0, depth - d_max))
    return start, min(depth, start + d_max)


def training_visibility_slab(depth: int, d_max: int, target_masks: Sequence[torch.Tensor],
                             dialogue_id: str, epoch: int = 0, seed: int = 0) -> tuple[int, int]:
    """TRAIN-ONLY GT policy maximizing the count of visible window targets.

    Ties are deterministic for (dialogue, epoch, seed), which allows epoch-wise
    variety without making worker scheduling affect sampling.
    """
    if depth <= d_max: return 0, depth
    candidates=[]; best=-1
    for start in range(depth-d_max+1):
        score=sum(bool(mask[start:start+d_max].any()) for mask in target_masks)
        if score>best: candidates=[start]; best=score
        elif score==best: candidates.append(start)
    digest=hashlib.sha256(f"{dialogue_id}:{epoch}:{seed}".encode()).digest()
    start=candidates[int.from_bytes(digest[:8],"little")%len(candidates)]
    return start,start+d_max


def evaluation_slabs(depth: int, d_max: int, policy: str = "centre") -> list[tuple[int, int]]:
    """Evaluation policies deliberately consume no target annotation."""
    if policy == "centre":
        return [centre_slab(depth, d_max)]
    if policy == "sliding":
        return [(start, min(depth, start + d_max)) for start in range(0, depth, d_max)]
    raise ValueError(f"unknown evaluation slab policy: {policy}")
