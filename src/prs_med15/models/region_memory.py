"""Persistent per-scan entity state and compact region tokens."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Literal
import torch
from torch import nn
import torch.nn.functional as F


@dataclass
class EntityEntry:
    scan_id: str; entity_id: str; version: int; source_turn: str
    seg_state: torch.Tensor; mask: torch.Tensor; pooled_feature: torch.Tensor
    geometry: Any; confidence: float
    provenance: Literal["oracle","predicted"]


class MaskedRegionPool(nn.Module):
    def __init__(self, feature_dim: int) -> None:
        super().__init__()
        self.fallback = nn.Parameter(torch.zeros(feature_dim))
        nn.init.normal_(self.fallback, std=.02)

    def forward(self, features: torch.Tensor, mask: torch.Tensor, valid_mask: torch.Tensor | None = None):
        """Pool [D,h,w,C] with mask on any spatial grid; returns vector, confidence flag."""
        depth, height, width, _ = features.shape
        target_size = (depth, height, width)
        weights = F.interpolate(
            mask.float().reshape(1, 1, *mask.shape[-3:]),
            size=target_size,
            mode="trilinear",
            align_corners=False,
        )[0, 0]
        if valid_mask is not None:
            valid = F.interpolate(
                valid_mask.float().reshape(1, 1, *valid_mask.shape[-3:]),
                size=target_size,
                mode="nearest",
            )[0, 0]
            weights = weights * valid
        denominator = weights.sum()
        pooled = (features * weights[..., None]).sum((0, 1, 2))
        pooled = pooled / denominator.clamp_min(1e-8)
        empty = denominator <= 0
        return torch.where(empty, self.fallback.to(pooled), pooled), float(not bool(empty))


class RegionTokenizer(nn.Module):
    def __init__(self, feature_dim: int, seg_dim: int, d_llm: int, tokens_per_entity: int = 4,
                 max_scans: int = 16, max_entities: int = 256, max_versions: int = 32) -> None:
        super().__init__(); self.tokens_per_entity=tokens_per_entity
        self.proj=nn.Sequential(nn.Linear(feature_dim+seg_dim,d_llm),nn.GELU(),nn.Linear(d_llm,tokens_per_entity*d_llm))
        self.scan=nn.Embedding(max_scans,d_llm); self.entity=nn.Embedding(max_entities,d_llm)
        self.version=nn.Embedding(max_versions,d_llm); self.provenance=nn.Embedding(2,d_llm)
        self.position=nn.Parameter(torch.randn(tokens_per_entity,d_llm)*.02)

    @staticmethod
    def _index(alias: str) -> int:
        digits="".join(x for x in alias if x.isdigit()); return max(0,int(digits or 1)-1)

    def forward(self, entry: EntityEntry) -> torch.Tensor:
        device=entry.pooled_feature.device
        base=self.proj(torch.cat((entry.pooled_feature,entry.seg_state.to(device)))).reshape(self.tokens_per_entity,-1)
        meta=self.scan.weight[self._index(entry.scan_id)]+self.entity.weight[self._index(entry.entity_id)]
        provenance_index = int(entry.provenance == "predicted")
        meta=meta+self.version.weight[min(entry.version,self.version.num_embeddings-1)]+self.provenance.weight[provenance_index]
        return base+meta+self.position


class EntityRegistry:
    def __init__(self) -> None:
        self._entries: dict[str, dict[str, EntityEntry]] = {}

    def contains(self, scan_id: str, entity_id: str) -> bool:
        return entity_id in self._entries.get(scan_id, {})
    def create(self, entry: EntityEntry) -> EntityEntry:
        space=self._entries.setdefault(entry.scan_id,{})
        if entry.entity_id in space: raise ValueError("entity already exists; use update")
        if entry.version != 1: raise ValueError("new entity version must be 1")
        space[entry.entity_id]=entry; return entry
    def update(self, scan_id: str, entity_id: str, entry: EntityEntry) -> EntityEntry:
        old=self.get(scan_id,entity_id)
        if entry.scan_id!=scan_id or entry.entity_id!=entity_id: raise ValueError("update cannot replace a different entity")
        if entry.version!=old.version+1: raise ValueError("update version must increment by one")
        self._entries[scan_id][entity_id]=entry; return entry
    def get(self, scan_id: str, entity_id: str) -> EntityEntry:
        try: return self._entries[scan_id][entity_id]
        except KeyError as exc: raise KeyError(f"unknown entity {scan_id}:{entity_id}") from exc
