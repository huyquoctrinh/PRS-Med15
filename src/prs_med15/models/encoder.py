"""One shared 2D backbone applied slice-wise to native 2D and 3D inputs."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F


@dataclass
class FeatureBundle:
    dense: torch.Tensor  # [B,D,h,w,C]
    skips: list[torch.Tensor]
    valid_mask: torch.Tensor  # [B,D,h,w]
    patch_geometry: Any = None
    transforms: list[Any] = field(default_factory=list)


class SliceEncoder(nn.Module, ABC):
    @abstractmethod
    def forward(self, images: torch.Tensor, valid_mask: torch.Tensor | None = None,
                transforms: list[Any] | None = None) -> FeatureBundle:
        """Encode [B,C,D,H,W] with validity [B,D,H,W]."""


def interpolate_2d_position_embedding(position: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    """Interpolate BCHW or BHWC absolute embeddings without assuming a checkpoint grid."""
    if position.ndim != 4:
        raise ValueError("absolute position embedding must be rank four")
    is_bhwc = position.shape[-1] > position.shape[1]
    value = position.permute(0, 3, 1, 2) if is_bhwc else position
    value = F.interpolate(value.float(), size=size, mode="bicubic", align_corners=False).to(position.dtype)
    return value.permute(0, 2, 3, 1) if is_bhwc else value


def interpolate_relative_position(position: torch.Tensor, length: int) -> torch.Tensor:
    if position.ndim != 2:
        raise ValueError("relative position table must be [L,C]")
    return F.interpolate(position.T[None].float(), size=length, mode="linear", align_corners=False)[0].T.to(position.dtype)


def _patch_valid(mask: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    b, d, h, w = mask.shape
    value = F.adaptive_avg_pool2d(mask.reshape(b * d, 1, h, w).float(), size)
    return value.reshape(b, d, *size) > 0


class TinyConvEncoder(SliceEncoder):
    """Small CPU test encoder preserving the production tensor contract."""
    def __init__(self, in_channels: int = 3, feature_dim: int = 16, patch_size: int = 4) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.patch_size = patch_size
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, feature_dim, 3, padding=1), nn.GELU(),
            nn.Conv2d(feature_dim, feature_dim, patch_size, stride=patch_size),
        )

    def forward(self, images: torch.Tensor, valid_mask: torch.Tensor | None = None,
                transforms: list[Any] | None = None) -> FeatureBundle:
        if images.ndim != 5:
            raise ValueError("images must be [B,C,D,H,W]")
        b, c, d, h, w = images.shape
        valid_mask = (torch.ones((b, d, h, w), dtype=torch.bool, device=images.device)
                      if valid_mask is None else valid_mask.bool())
        slice_valid = valid_mask.flatten(2).any(-1)
        flat = images.permute(0, 2, 1, 3, 4).reshape(b * d, c, h, w)
        flat_mask = valid_mask.reshape(b * d, 1, h, w)
        indices = torch.where(slice_valid.reshape(-1))[0]
        oh, ow = h // self.patch_size, w // self.patch_size
        output = images.new_zeros((b * d, self.feature_dim, oh, ow))
        if indices.numel():
            output[indices] = self.net(flat[indices] * flat_mask[indices])
        patch_valid = _patch_valid(valid_mask, (oh, ow))
        output = output.reshape(b, d, self.feature_dim, oh, ow).permute(0, 1, 3, 4, 2)
        output = output * patch_valid.unsqueeze(-1)
        return FeatureBundle(output, [], patch_valid,
                             {"patch_size_pixels": (self.patch_size, self.patch_size)}, transforms or [])


class MedSamSliceEncoder(SliceEncoder):
    """MedSAM ViT-B wrapper with 512-grid interpolation and slice micro-batching.

    The basic wrapper is adapted from PRS-Med/prs_med/models/sam_encoder.py.
    """
    def __init__(self, model_id: str = "wanglab/medsam-vit-base", image_size: int = 512,
                 microbatch: int = 8, freeze: bool = True, cache_dir: str | None = None,
                 local_files_only: bool = True, model: nn.Module | None = None) -> None:
        super().__init__()
        if model is None:
            from transformers import SamModel
            model = SamModel.from_pretrained(
                model_id,
                cache_dir=cache_dir,
                local_files_only=local_files_only,
            )
        self.image_encoder = getattr(model, "vision_encoder", model)
        self.image_size = image_size
        self.microbatch = microbatch
        self._adapt_grid(image_size)
        self.train_from_layer: int | None = None
        self.checkpoint_trainable = True
        if freeze:
            self.requires_grad_(False)
            self.eval()

    def set_trainable_top(self, first_layer: int, checkpoint: bool = True) -> None:
        """Unfreeze layers[first_layer:] + neck; earlier layers stay frozen and run without grad."""
        self.requires_grad_(False)
        for module in list(self.image_encoder.layers[first_layer:]) + [self.image_encoder.neck]:
            module.requires_grad_(True)
        self.train_from_layer = first_layer
        self.checkpoint_trainable = checkpoint

    def _encode(self, pixels: torch.Tensor) -> torch.Tensor:
        """Run the ViT; with a trainable top, the frozen bottom is computed under no_grad."""
        encoder = self.image_encoder
        if self.train_from_layer is None or not torch.is_grad_enabled():
            if self.train_from_layer is None:
                frozen = not any(p.requires_grad for p in encoder.parameters())
            else:
                frozen = True
            with torch.set_grad_enabled(torch.is_grad_enabled() and not frozen):
                result = encoder(pixels)
            return result[0] if isinstance(result, (tuple, list)) else getattr(result, "last_hidden_state", result)
        from torch.utils.checkpoint import checkpoint
        with torch.no_grad():
            hidden = encoder.patch_embed(pixels)
            if encoder.pos_embed is not None:
                hidden = hidden + encoder.pos_embed
            for layer in encoder.layers[:self.train_from_layer]:
                hidden = layer(hidden)
        hidden = hidden.detach()
        for layer in encoder.layers[self.train_from_layer:]:
            if self.checkpoint_trainable:
                hidden = checkpoint(layer, hidden, use_reentrant=False)
            else:
                hidden = layer(hidden)
            if isinstance(hidden, (tuple, list)):
                hidden = hidden[0]
        return encoder.neck(hidden)

    def _adapt_grid(self, image_size: int) -> None:
        patch = getattr(getattr(self.image_encoder, "config", None), "patch_size", 16)
        target = (image_size // patch, image_size // patch)
        patch_embed = getattr(self.image_encoder, "patch_embed", None)
        if patch_embed is not None and hasattr(patch_embed, "image_size"):
            patch_embed.image_size = (image_size, image_size)
        if hasattr(self.image_encoder, "pos_embed") and self.image_encoder.pos_embed is not None:
            old = self.image_encoder.pos_embed
            self.image_encoder.pos_embed = nn.Parameter(interpolate_2d_position_embedding(old, target).contiguous(),
                                                        requires_grad=old.requires_grad)
        layers = getattr(self.image_encoder, "layers", [])
        for layer in layers:
            attention = getattr(layer, "attn", getattr(layer, "attention", None))
            if attention is None:
                continue
            # Windowed-attention layers keep their (2*window-1) tables; only global layers
            # (window_size == 0) attend over the full patch grid and need resizing.
            if getattr(layer, "window_size", 0):
                continue
            for name, side in (("rel_pos_h", target[0]), ("rel_pos_w", target[1])):
                if hasattr(attention, name):
                    old = getattr(attention, name)
                    length = 2 * side - 1
                    setattr(attention, name, nn.Parameter(interpolate_relative_position(old, length).contiguous(),
                                                          requires_grad=old.requires_grad))
        if hasattr(self.image_encoder, "config"):
            self.image_encoder.config.image_size = image_size

    # SAM/MedSAM pixel statistics on the 0-255 scale (SamProcessor defaults).
    PIXEL_MEAN = (123.675, 116.28, 103.53)
    PIXEL_STD = (58.395, 57.12, 57.375)

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Map preprocessed [0,1] slices to MedSAM's expected input statistics."""
        mean = x.new_tensor(self.PIXEL_MEAN).view(1, 3, 1, 1)
        std = x.new_tensor(self.PIXEL_STD).view(1, 3, 1, 1)
        return (x * 255.0 - mean) / std

    def forward(self, images: torch.Tensor, valid_mask: torch.Tensor | None = None,
                transforms: list[Any] | None = None) -> FeatureBundle:
        if images.ndim != 5:
            raise ValueError("images must be [B,C,D,H,W]")
        b, c, d, h, w = images.shape
        if (h, w) != (self.image_size, self.image_size):
            raise ValueError(f"MedSAM expects {self.image_size}x{self.image_size}")
        valid_mask = (torch.ones((b, d, h, w), dtype=torch.bool, device=images.device)
                      if valid_mask is None else valid_mask.bool())
        flat = images.permute(0, 2, 1, 3, 4).reshape(b * d, c, h, w)
        flat_valid = valid_mask.reshape(b * d, h, w)
        indices = torch.where(flat_valid.flatten(1).any(-1))[0]
        encoded: list[torch.Tensor] = []
        for chunk in indices.split(self.microbatch):
            # Normalise, then zero padded pixels so padding content never depends on statistics.
            value = self._encode(self.normalize(flat[chunk]) * flat_valid[chunk, None])
            if value.ndim == 4 and value.shape[-1] > value.shape[1]:
                value = value.permute(0, 3, 1, 2)
            encoded.append(value)
        if encoded:
            values = torch.cat(encoded)
            _, channels, oh, ow = values.shape
        else:
            patch = getattr(getattr(self.image_encoder, "config", None), "patch_size", 16)
            channels, oh, ow = 256, h // patch, w // patch
            values = flat.new_empty((0, channels, oh, ow))
        output = flat.new_zeros((b * d, channels, oh, ow))
        if indices.numel():
            output = output.index_put((indices,), values.to(output.dtype))
        patch_valid = _patch_valid(valid_mask, (oh, ow))
        dense = output.reshape(b, d, channels, oh, ow).permute(0, 1, 3, 4, 2)
        dense = dense * patch_valid.unsqueeze(-1)
        return FeatureBundle(dense, [], patch_valid,
                             {"patch_size_pixels": (h / oh, w / ow)}, transforms or [])
