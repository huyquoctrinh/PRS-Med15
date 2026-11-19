"""Partial MedSAM fine-tuning: frozen bottom gets no grad, top + neck do, output equals the plain forward."""
import torch
from transformers import SamVisionConfig
from transformers.models.sam.modeling_sam import SamVisionEncoder

from prs_med15.models.encoder import MedSamSliceEncoder
from prs_med15.training.trainer import _new_module_state


def _encoder() -> MedSamSliceEncoder:
    torch.manual_seed(0)
    config = SamVisionConfig(hidden_size=32, num_hidden_layers=4, num_attention_heads=2, image_size=64,
                             patch_size=16, output_channels=16, window_size=2, global_attn_indexes=[1, 3],
                             mlp_dim=64)
    vision = SamVisionEncoder(config)
    with torch.no_grad():  # random init leaves some tensors zero; make every path carry signal
        for parameter in vision.parameters():
            parameter.normal_(0, 0.2)
    return MedSamSliceEncoder(image_size=64, microbatch=2, model=vision)


def test_trainable_top_matches_frozen_forward_and_routes_grads():
    enc = _encoder()
    images, valid = torch.rand(1, 3, 3, 64, 64), torch.ones(1, 3, 64, 64, dtype=torch.bool)
    with torch.no_grad():
        reference = enc(images, valid).dense
    enc.set_trainable_top(2, checkpoint=True)
    dense = enc(images, valid).dense
    assert torch.allclose(dense, reference, atol=1e-5)
    dense.square().mean().backward()
    layers = enc.image_encoder.layers
    assert all(p.grad is None for p in layers[:2].parameters())
    assert enc.image_encoder.pos_embed.grad is None
    assert all(p.grad is not None and p.grad.abs().sum() > 0 for p in layers[2:].parameters())
    assert all(p.grad is not None for p in enc.image_encoder.neck.parameters())


def test_trainable_encoder_params_are_checkpointed():
    model = torch.nn.Module()
    model.encoder = _encoder()
    model.encoder.set_trainable_top(3)
    keys = [k for k in _new_module_state(model) if k.startswith("encoder.")]
    assert keys and all(".layers.3." in k or ".neck." in k for k in keys)
