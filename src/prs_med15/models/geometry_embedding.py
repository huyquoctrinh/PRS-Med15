"""Physical geometry features for compact and dense visual tokens."""
from __future__ import annotations

import math
import torch
from torch import nn


def sincos_3d(coords: torch.Tensor, dim: int) -> torch.Tensor:
    """Return a 3-D sine/cosine PE for normalized ``[..., 3]`` coordinates."""
    if coords.shape[-1] != 3:
        raise ValueError("coords must end in (z,y,x)")
    per_axis = max(2, 2 * (dim // 6))
    half = per_axis // 2
    freq = torch.exp(-math.log(10000.0) * torch.arange(half, device=coords.device) /
                     max(half - 1, 1)).to(coords.dtype)
    parts = []
    for axis in range(3):
        angle = coords[..., axis, None] * (2 * math.pi) * freq
        parts.extend((angle.sin(), angle.cos()))
    out = torch.cat(parts, -1)
    if out.shape[-1] < dim:
        out = torch.nn.functional.pad(out, (0, dim - out.shape[-1]))
    return out[..., :dim]


class GeometryEmbedding(nn.Module):
    """MLP over centre/extent, orientation, modality, kind and unknown flag."""
    def __init__(self, out_dim: int, orientation_dim: int = 6, modalities: int = 4,
                 input_kinds: int = 3, hidden_dim: int | None = None) -> None:
        super().__init__()
        hidden_dim = hidden_dim or out_dim
        self.orientation_dim = orientation_dim
        self.modality = nn.Embedding(modalities, 8)
        self.input_kind = nn.Embedding(input_kinds, 8)
        self.mlp = nn.Sequential(nn.Linear(6 + orientation_dim + 8 + 8 + 1, hidden_dim),
                                 nn.GELU(), nn.Linear(hidden_dim, out_dim))

    def forward(self, centre_mm: torch.Tensor, extent_mm: torch.Tensor,
                orientation: torch.Tensor, modality_id: torch.Tensor,
                input_kind_id: torch.Tensor, unknown_geometry: torch.Tensor) -> torch.Tensor:
        unknown = unknown_geometry.to(centre_mm.dtype).unsqueeze(-1)
        known = 1.0 - unknown
        # Unknown coordinates are explicitly zero, never inferred from array indices.
        centre = centre_mm * known
        extent = extent_mm * known
        orient = orientation.to(centre.dtype) * known
        x = torch.cat((centre, extent, orient, self.modality(modality_id),
                       self.input_kind(input_kind_id), unknown), -1)
        return self.mlp(x)
