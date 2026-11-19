"""[SEG]-conditioned slice-wise MedSAM decoders."""
from __future__ import annotations

from contextlib import nullcontext

import torch
from torch import nn
import torch.nn.functional as F


def _no_autocast(device: torch.device):
    if device.type in {"cpu", "cuda"}:
        return torch.autocast(device_type=device.type, enabled=False)
    return nullcontext()


class TinyMaskDecoder(nn.Module):
    """Small CPU decoder with the production decoder interface."""

    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int,
        prompt_dim: int | None = None,
        sparse_tokens: int = 2,
        microbatch: int = 8,
    ) -> None:
        super().__init__()
        prompt_dim = prompt_dim or feature_dim
        self.sparse_tokens = sparse_tokens
        self.microbatch = microbatch
        self.query = nn.Sequential(
            nn.Linear(hidden_dim, prompt_dim),
            nn.GELU(),
            nn.Linear(prompt_dim, sparse_tokens * feature_dim),
        )
        self.refine = nn.Sequential(
            nn.Conv2d(feature_dim + 1, feature_dim, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(feature_dim, 1, 1),
        )

    def forward(
        self,
        features: torch.Tensor,
        seg_state: torch.Tensor,
        output_size: tuple[int, int],
        valid_mask: torch.Tensor | None = None,
        prior_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if features.ndim != 4:
            raise ValueError("features must be [D,C,h,w]")
        with _no_autocast(features.device):
            values = features.float()
            depth, _, height, width = values.shape
            query = self.query(seg_state.float().reshape(1, -1))
            query = query.reshape(self.sparse_tokens, -1).mean(0)
            values = values + query[None, :, None, None]
            if prior_mask is None:
                prior = torch.zeros((depth, 1, height, width), device=values.device)
            else:
                prior = F.interpolate(
                    prior_mask.float().reshape(depth, 1, *prior_mask.shape[-2:]),
                    (height, width),
                    mode="bilinear",
                    align_corners=False,
                )
            chunks = []
            for start in range(0, depth, self.microbatch):
                stop = start + self.microbatch
                chunks.append(self.refine(torch.cat((values[start:stop], prior[start:stop]), dim=1)))
            logits = F.interpolate(
                torch.cat(chunks),
                output_size,
                mode="bilinear",
                align_corners=False,
            )
            if valid_mask is not None:
                logits = logits.masked_fill(~valid_mask[:, None].bool(), -20.0)
            return logits


class MedSAMMaskDecoder(nn.Module):
    """Wrap HF SAM prompt/mask modules while preserving checkpoint PE weights."""

    def __init__(
        self,
        prompt_encoder: nn.Module,
        mask_decoder: nn.Module,
        shared_image_embedding: nn.Module,
        hidden_dim: int,
        sam_dim: int = 256,
        sparse_tokens: int = 2,
        microbatch: int = 8,
        extra_tokens: int = 0,
        film: bool = False,
    ) -> None:
        super().__init__()
        self.extra_tokens = extra_tokens
        # Entity-conditioning reinforcement (2026-11-18 diagnosis: the decoder ignored the [SEG] query).
        # Both paths read a LayerNorm'd seg state; FiLM is zero-initialised and the extra tokens are
        # appended, so a checkpoint without them starts from (nearly) the same function.
        if extra_tokens or film:
            self.query_norm = nn.LayerNorm(hidden_dim)
        if extra_tokens:
            self.query_extra = nn.Sequential(
                nn.Linear(hidden_dim, sam_dim), nn.GELU(), nn.Linear(sam_dim, extra_tokens * sam_dim))
        self.film = None
        if film:
            self.film = nn.Sequential(nn.Linear(hidden_dim, sam_dim), nn.GELU(), nn.Linear(sam_dim, 2 * sam_dim))
            nn.init.zeros_(self.film[-1].weight)
            nn.init.zeros_(self.film[-1].bias)
        self.prompt_encoder = prompt_encoder
        self.mask_decoder = mask_decoder
        self.shared_image_embedding = shared_image_embedding
        self.sparse_tokens = sparse_tokens
        self.microbatch = microbatch
        self.query = nn.Sequential(
            nn.Linear(hidden_dim, sam_dim),
            nn.GELU(),
            nn.Linear(sam_dim, sparse_tokens * sam_dim),
        )
        for module in self.query:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def image_positional_embeddings(
        self,
        height: int,
        width: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Evaluate SAM's learned random-Fourier PE on the actual feature grid."""
        y = (torch.arange(height, device=device, dtype=dtype) + 0.5) / height
        x = (torch.arange(width, device=device, dtype=dtype) + 0.5) / width
        y_grid, x_grid = torch.meshgrid(y, x, indexing="ij")
        coordinates = torch.stack((x_grid, y_grid), dim=-1)
        positional = self.shared_image_embedding(coordinates)
        return positional.permute(2, 0, 1).unsqueeze(0)

    def _dense_prompt(
        self,
        prior_mask: torch.Tensor | None,
        depth: int,
        height: int,
        width: int,
    ) -> torch.Tensor:
        if prior_mask is None:
            return self.prompt_encoder.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(
                depth,
                -1,
                height,
                width,
            )
        mask_input = F.interpolate(
            prior_mask.float().reshape(depth, 1, *prior_mask.shape[-2:]),
            size=(4 * height, 4 * width),
            mode="bilinear",
            align_corners=False,
        )
        dense = self.prompt_encoder.mask_embed(mask_input)
        if dense.shape[-2:] != (height, width):
            raise ValueError(
                "SAM mask_embed returned an incompatible grid: "
                f"{tuple(dense.shape[-2:])} != {(height, width)}"
            )
        return dense

    def forward(
        self,
        features: torch.Tensor,
        seg_state: torch.Tensor,
        output_size: tuple[int, int],
        valid_mask: torch.Tensor | None = None,
        prior_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if features.ndim != 4:
            raise ValueError("features must be [D,C,h,w]")
        with _no_autocast(features.device):
            values = features.float()
            depth, channels, height, width = values.shape
            state = seg_state.float().reshape(1, -1)
            sparse = self.query(state).reshape(1, 1, self.sparse_tokens, channels)
            if self.extra_tokens or self.film is not None:
                normed = self.query_norm(state)
            if self.extra_tokens:
                extra = self.query_extra(normed).reshape(1, 1, self.extra_tokens, channels)
                sparse = torch.cat((sparse, extra), dim=2)
            if self.film is not None:
                gamma, beta = self.film(normed).reshape(2, channels)
                values = values * (1 + gamma[None, :, None, None]) + beta[None, :, None, None]
            sparse = sparse.expand(depth, -1, -1, -1)
            dense = self._dense_prompt(prior_mask, depth, height, width).float()
            positional = self.image_positional_embeddings(
                height,
                width,
                values.device,
                values.dtype,
            ).expand(depth, -1, -1, -1)
            outputs = []
            for start in range(0, depth, self.microbatch):
                stop = start + self.microbatch
                masks, _ = self.mask_decoder(
                    image_embeddings=values[start:stop],
                    image_positional_embeddings=positional[start:stop],
                    sparse_prompt_embeddings=sparse[start:stop],
                    dense_prompt_embeddings=dense[start:stop],
                    multimask_output=False,
                )
                outputs.append(masks[:, 0, 0].unsqueeze(1))
            logits = F.interpolate(
                torch.cat(outputs),
                output_size,
                mode="bilinear",
                align_corners=False,
            )
            if valid_mask is not None:
                logits = logits.masked_fill(~valid_mask[:, None].bool(), -20.0)
            return logits


def load_medsam_mask_decoder(
    model_id: str = "wanglab/medsam-vit-base",
    *,
    cache_dir: str | None = None,
    local_files_only: bool = True,
    **kwargs,
) -> MedSAMMaskDecoder:
    from transformers import SamModel

    sam = SamModel.from_pretrained(
        model_id,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
    )
    return MedSAMMaskDecoder(
        sam.prompt_encoder,
        sam.mask_decoder,
        sam.shared_image_embedding,
        **kwargs,
    )
