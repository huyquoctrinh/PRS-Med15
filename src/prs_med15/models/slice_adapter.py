"""Depth attention plus a shared spatial residual block."""
from __future__ import annotations

import torch
from torch import nn


class SliceAdapter(nn.Module):
    def __init__(self, dim: int, heads: int = 8, layers: int = 2, alpha_init: float = 0.0) -> None:
        super().__init__()
        if dim % heads:
            raise ValueError("dim must be divisible by heads")
        self.depth_layers = nn.ModuleList([
            nn.MultiheadAttention(dim, heads, batch_first=True) for _ in range(layers)
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(dim) for _ in range(layers)])
        self.spatial = nn.Sequential(nn.Conv2d(dim, dim, 3, padding=1, groups=dim),
                                     nn.GELU(), nn.Conv2d(dim, dim, 1))
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))

    @staticmethod
    def depth_encoding(positions: torch.Tensor, dim: int, scale_mm: float = 64.0) -> torch.Tensor:
        """Sinusoidal encoding of slice positions [B,D] (mm along S when known, else index)."""
        half = dim // 2
        freqs = torch.exp(-torch.arange(half, device=positions.device, dtype=torch.float32)
                          * (torch.log(torch.tensor(10000.0)) / max(half - 1, 1)))
        angles = positions.float().unsqueeze(-1) / scale_mm * 2 * torch.pi * freqs
        enc = torch.cat([angles.sin(), angles.cos()], dim=-1)
        if enc.shape[-1] < dim:
            enc = torch.nn.functional.pad(enc, (0, dim - enc.shape[-1]))
        return enc

    def forward(self, features: torch.Tensor, valid_mask: torch.Tensor,
                depth_positions: torch.Tensor | None = None) -> torch.Tensor:
        """depth_positions: [B,D] slice positions in mm (or slice index when geometry is unknown).

        Depth attention is otherwise permutation-invariant, so slice order must be injected here.
        """
        if features.ndim != 5 or valid_mask.shape != features.shape[:-1]:
            raise ValueError("expected features [B,D,H,W,C] and matching valid mask")
        b, d, h, w, c = features.shape
        if depth_positions is None:
            depth_positions = torch.arange(d, device=features.device).expand(b, d)
        pe = self.depth_encoding(depth_positions, c).to(features.dtype)  # [B,D,C]
        pe = pe[:, None, None].expand(b, h, w, d, c).reshape(b * h * w, d, c)
        x = features.permute(0, 2, 3, 1, 4).reshape(b * h * w, d, c)
        mask = valid_mask.permute(0, 2, 3, 1).reshape(b * h * w, d)
        active = mask.any(-1)
        adapted = torch.zeros_like(x)
        if active.any():
            y = x[active]
            pos = pe[active]
            padding = ~mask[active]
            for attention, norm in zip(self.depth_layers, self.norms, strict=True):
                qk = norm(y) + pos
                update, _ = attention(qk, qk, norm(y), key_padding_mask=padding,
                                      need_weights=False)
                y = y + update
                y = y.masked_fill(padding.unsqueeze(-1), 0)
            adapted[active] = y
        adapted = adapted.reshape(b, h, w, d, c).permute(0, 3, 4, 1, 2)
        spatial = self.spatial(adapted.reshape(b * d, c, h, w))
        spatial = spatial.reshape(b, d, c, h, w).permute(0, 1, 3, 4, 2)
        spatial = spatial * valid_mask.unsqueeze(-1)
        return features + self.alpha * spatial
