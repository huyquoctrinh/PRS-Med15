"""Instruction-conditioned cross attention over uncompressed dense features."""
from __future__ import annotations
import torch
from torch import nn
from .geometry_embedding import sincos_3d


class DetailRetriever(nn.Module):
    def __init__(self, feature_dim: int, d_r: int, d_llm: int, layers: int = 2, heads: int = 8) -> None:
        super().__init__()
        self.key = nn.Linear(feature_dim, d_r); self.value = nn.Linear(feature_dim, d_r)
        self.layers = nn.ModuleList([nn.MultiheadAttention(d_r, heads, batch_first=True) for _ in range(layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(d_r) for _ in range(layers)])
        self.out = nn.Linear(d_r, d_llm)

    def prepare(self, dense: torch.Tensor, valid_mask: torch.Tensor, query_dim: int):
        """Dialogue-invariant keys/values/PE; reuse across turns of the same scan."""
        b,d,h,w,c = dense.shape
        z,y,x = torch.meshgrid(torch.linspace(0,1,d,device=dense.device), torch.linspace(0,1,h,device=dense.device),
                               torch.linspace(0,1,w,device=dense.device), indexing="ij")
        pe = sincos_3d(torch.stack((z,y,x),-1), query_dim).reshape(1,d*h*w,-1)
        flat, valid = dense.reshape(b,-1,c), valid_mask.reshape(b,-1).bool()
        key, value = self.key(flat) + pe.to(flat.dtype), self.value(flat)
        return key, value, valid, valid.any(-1)

    def attend(self, queries: torch.Tensor, prepared) -> torch.Tensor:
        key, value, valid, active = prepared
        out = queries
        result = torch.zeros_like(out)
        if active.any():
            yq = out[active]
            for attn,norm in zip(self.layers,self.norms,strict=True):
                update,_ = attn(norm(yq), key[active], value[active], key_padding_mask=~valid[active], need_weights=False)
                yq = yq + update
            result[active] = yq
        return self.out(result)

    def forward(self, queries: torch.Tensor, dense: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        return self.attend(queries, self.prepare(dense, valid_mask, queries.shape[-1]))
