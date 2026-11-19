"""Masked spatial overview pooling with an explicit depth budget policy."""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any
import torch
from torch import nn
import torch.nn.functional as F

from .geometry_embedding import GeometryEmbedding


def depth_after_budget(depth: int, budget: int, cells_per_slice: int = 16) -> tuple[int, int]:
    """Shared dataset/model rule: repeated adjacent-pair pooling until within budget."""
    rounds = 0
    while depth * cells_per_slice > budget and depth > 1:
        depth = (depth + 1) // 2
        rounds += 1
    return depth, rounds


def global_token_count(valid_depth: int, budget: int, cells_per_slice: int = 16) -> int:
    return depth_after_budget(valid_depth, budget, cells_per_slice)[0] * cells_per_slice


@dataclass
class GlobalTokenOutput:
    tokens: torch.Tensor
    token_mask: torch.Tensor
    cell_geometry: dict[str, torch.Tensor]
    policy: dict[str, Any]


class GlobalTokenPooler(nn.Module):
    def __init__(self, feature_dim: int, d_llm: int, budget: int = 512,
                 cells: tuple[int, int] = (4, 4), geometry_dim: int | None = None) -> None:
        super().__init__()
        self.budget, self.cells = budget, cells
        gd = geometry_dim or feature_dim
        self.geometry = GeometryEmbedding(gd)
        self.geometry_proj = nn.Linear(gd, feature_dim) if gd != feature_dim else nn.Identity()
        self.project = nn.Linear(feature_dim, d_llm)

    @staticmethod
    def _pool_spatial(x: torch.Tensor, valid: torch.Tensor, cells: tuple[int, int]):
        b, d, h, w, c = x.shape
        numer = F.adaptive_avg_pool2d((x * valid[..., None]).permute(0,1,4,2,3).reshape(b*d,c,h,w), cells)
        denom = F.adaptive_avg_pool2d(valid.reshape(b*d,1,h,w).float(), cells)
        pooled = (numer / denom.clamp_min(1e-8)).reshape(b,d,c,*cells).permute(0,1,3,4,2)
        cell_has_pixels = denom.reshape(b,d,*cells) > 0
        pooled = pooled.masked_fill(~cell_has_pixels[..., None], 0)
        # Token validity is per SLICE (the dataset reserves cells_per_slice slots for every valid
        # slice via global_token_count). Cells of a valid slice that fall entirely in letterbox
        # padding stay as zero-feature tokens (they still carry their geometry embedding);
        # masking them per cell made the token count disagree with the reserved slots.
        slice_valid = valid.reshape(b, d, -1).any(-1)
        mask = slice_valid[:, :, None, None].expand(b, d, *cells)
        return pooled, mask

    @staticmethod
    def _pair_depth(x: torch.Tensor, mask: torch.Tensor):
        if x.shape[1] % 2:
            x = torch.cat((x, torch.zeros_like(x[:, :1])), 1)
            mask = torch.cat((mask, torch.zeros_like(mask[:, :1])), 1)
        b, d = x.shape[:2]
        xr, mr = x.reshape(b,d//2,2,*x.shape[2:]), mask.reshape(b,d//2,2,*mask.shape[2:])
        den = mr.sum(2)
        out = (xr * mr[..., None]).sum(2) / den.clamp_min(1)[..., None]
        return out.masked_fill(~den.bool()[..., None], 0), den.bool()

    def forward(self, dense: torch.Tensor, valid_mask: torch.Tensor,
                geometry: dict[str, torch.Tensor] | None = None) -> GlobalTokenOutput:
        x, mask = self._pool_spatial(dense, valid_mask.bool(), self.cells)
        original_depth, rounds = x.shape[1], 0
        while x.shape[1] * (self.cells[0] * self.cells[1]) > self.budget and x.shape[1] > 1:
            x, mask = self._pair_depth(x, mask); rounds += 1
        b,d,cy,cx,c = x.shape
        z = (torch.arange(d, device=x.device, dtype=x.dtype) + .5) / max(d,1)
        yy = (torch.arange(cy, device=x.device, dtype=x.dtype) + .5) / cy
        xx = (torch.arange(cx, device=x.device, dtype=x.dtype) + .5) / cx
        zz, gy, gx = torch.meshgrid(z, yy, xx, indexing="ij")
        centre = torch.stack((zz,gy,gx),-1).expand(b,-1,-1,-1,-1).clone()
        extent = x.new_tensor((1/max(d,1),1/cy,1/cx)).expand_as(centre).clone()
        unknown = torch.ones((b,d,cy,cx), dtype=torch.bool, device=x.device)
        orientation = x.new_zeros((b,d,cy,cx,6)); orientation[..., :3] = 1
        modality = torch.zeros((b,d,cy,cx), dtype=torch.long, device=x.device)
        kind = torch.zeros_like(modality)
        if geometry:
            # Coordinates are considered known only when explicitly supplied.
            if "centre_mm" in geometry and "extent_mm" in geometry:
                centre = geometry["centre_mm"].to(x).reshape_as(centre)
                extent = geometry["extent_mm"].to(x).reshape_as(extent)
                unknown = geometry.get("unknown_geometry", torch.zeros_like(unknown)).to(x.device).bool()
                if unknown.ndim < 4: unknown = unknown.reshape(b,1,1,1).expand_as(mask)
            modality = geometry.get("modality_id", modality)
            kind = geometry.get("input_kind_id", kind)
            if not torch.is_tensor(modality): modality=torch.full_like(mask,int(modality),dtype=torch.long)
            if not torch.is_tensor(kind): kind=torch.full_like(mask,int(kind),dtype=torch.long)
            if modality.ndim<4: modality=modality.reshape(b,1,1,1).expand_as(mask)
            if kind.ndim<4: kind=kind.reshape(b,1,1,1).expand_as(mask)
        centre = centre.masked_fill(unknown[...,None], 0)
        extent = extent.masked_fill(unknown[...,None], 0)
        ge = self.geometry(centre, extent, orientation, modality, kind, unknown)
        x = x + self.geometry_proj(ge)
        return GlobalTokenOutput(self.project(x.reshape(b,-1,c)), mask.reshape(b,-1),
            {"centre_mm": centre.reshape(b,-1,3), "extent_mm": extent.reshape(b,-1,3),
             "unknown_geometry": unknown.reshape(b,-1)},
            {"spatial_cells": list(self.cells), "input_depth": original_depth,
             "output_depth": d, "depth_pair_rounds": rounds, "budget": self.budget})
