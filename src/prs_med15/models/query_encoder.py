"""Past-only question and region query construction."""
from __future__ import annotations
import torch
from torch import nn


class QueryEncoder(nn.Module):
    def __init__(self, embedding_table: nn.Module, d_llm: int, d_r: int = 512,
                 slots: int = 64, layers: int = 2, heads: int = 8) -> None:
        super().__init__()
        self.embedding_table = embedding_table
        self.slots = nn.Parameter(torch.randn(slots, d_r) * .02)
        self.question_proj = nn.Linear(d_llm, d_r)
        self.region_proj = nn.Linear(d_llm, d_r)
        self.type_embed = nn.Embedding(3, d_r)
        layer = nn.TransformerEncoderLayer(d_r, heads, 4*d_r, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers)

    def forward(self, question_ids: torch.Tensor, question_mask: torch.Tensor | None = None,
                region_entries: torch.Tensor | None = None,
                region_mask: torch.Tensor | None = None) -> torch.Tensor:
        """The narrow API intentionally has no answer ids or target-mask argument."""
        b, q = question_ids.shape
        question_mask = torch.ones_like(question_ids, dtype=torch.bool) if question_mask is None else question_mask.bool()
        question = self.question_proj(self.embedding_table(question_ids).detach().to(self.question_proj.weight.dtype)) + self.type_embed.weight[0]
        slot = self.slots[None].expand(b,-1,-1) + self.type_embed.weight[1]
        pieces, masks = [question, slot], [question_mask, torch.ones((b,slot.shape[1]),dtype=torch.bool,device=slot.device)]
        if region_entries is not None:
            pieces.append(self.region_proj(region_entries) + self.type_embed.weight[2])
            masks.append(torch.ones(region_entries.shape[:2],dtype=torch.bool,device=slot.device) if region_mask is None else region_mask.bool())
        x, valid = torch.cat(pieces,1), torch.cat(masks,1)
        y = self.encoder(x, src_key_padding_mask=~valid)
        return y[:, q:q+slot.shape[1]]
