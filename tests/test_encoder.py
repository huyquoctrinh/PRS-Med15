import torch

from prs_med15.models.encoder import (MedSamSliceEncoder, TinyConvEncoder,
                                      interpolate_2d_position_embedding,
                                      interpolate_relative_position)
from prs_med15.models.slice_adapter import SliceAdapter


def test_tiny_encoder_padding_invariance_and_invalid_slice_zero():
    torch.manual_seed(0)
    encoder = TinyConvEncoder(feature_dim=8, patch_size=4).eval()
    image = torch.randn(1, 3, 2, 8, 8)
    valid = torch.ones(1, 2, 8, 8, dtype=torch.bool)
    base = encoder(image, valid)
    padded = torch.zeros(1, 3, 3, 12, 12)
    padded[:, :, :2, :8, :8] = image
    padded_valid = torch.zeros(1, 3, 12, 12, dtype=torch.bool)
    padded_valid[:, :2, :8, :8] = True
    large = encoder(padded, padded_valid)
    assert torch.allclose(base.dense, large.dense[:, :2, :2, :2], atol=1e-6)
    assert not large.dense[:, 2].any()


def test_adapter_alpha_identity_and_d1():
    torch.manual_seed(0)
    adapter = SliceAdapter(8, heads=2, layers=2, alpha_init=0)
    x = torch.randn(2, 1, 3, 4, 8)
    valid = torch.ones(2, 1, 3, 4, dtype=torch.bool)
    assert torch.equal(adapter(x, valid), x)
    adapter.alpha.data.fill_(1)
    output = adapter(x, valid)
    assert output.shape == x.shape and torch.isfinite(output).all()


def test_medsam_embedding_interpolation_helpers():
    absolute = torch.randn(1, 4, 4, 12)
    assert interpolate_2d_position_embedding(absolute, (2, 3)).shape == (1, 2, 3, 12)
    relative = torch.randn(7, 3)
    assert interpolate_relative_position(relative, 3).shape == (3, 3)


def test_tiny_random_medsam_grid_without_weights():
    from transformers import SamVisionConfig
    from transformers.models.sam.modeling_sam import SamVisionEncoder
    config = SamVisionConfig(hidden_size=32, output_channels=16, num_hidden_layers=1,
                             num_attention_heads=4, mlp_dim=64, image_size=32,
                             patch_size=8, global_attn_indexes=[0], window_size=2)
    encoder = MedSamSliceEncoder(image_size=16, microbatch=2, freeze=True,
                                 model=SamVisionEncoder(config))
    image = torch.randn(1, 3, 1, 16, 16)
    bundle = encoder(image, torch.ones(1, 1, 16, 16, dtype=torch.bool))
    assert bundle.dense.shape == (1, 1, 2, 2, 16)


def test_medsam_windowed_rel_pos_untouched_and_global_resized():
    from transformers import SamVisionConfig
    from transformers.models.sam.modeling_sam import SamVisionEncoder
    config = SamVisionConfig(hidden_size=32, output_channels=16, num_hidden_layers=2,
                             num_attention_heads=2, image_size=64, patch_size=8,
                             global_attn_indexes=[1], window_size=2, mlp_dim=32)
    model = SamVisionEncoder(config)
    win_before = model.layers[0].attn.rel_pos_h.shape
    encoder = MedSamSliceEncoder(image_size=32, microbatch=2, freeze=True, model=model)
    assert model.layers[0].attn.rel_pos_h.shape == win_before  # windowed: 2*window-1
    assert model.layers[1].attn.rel_pos_h.shape[0] == 2 * (32 // 8) - 1  # global: full grid
    images = torch.rand(1, 3, 2, 32, 32)
    bundle = encoder(images)
    assert bundle.dense.shape[:4] == (1, 2, 4, 4)


def test_adapter_depth_order_matters():
    torch.manual_seed(0)
    adapter = SliceAdapter(dim=8, heads=2, layers=1, alpha_init=1.0)
    feats = torch.randn(1, 3, 2, 2, 8)
    valid = torch.ones(1, 3, 2, 2, dtype=torch.bool)
    out = adapter(feats, valid)
    out_perm = adapter(feats[:, [2, 1, 0]], valid)[:, [2, 1, 0]]
    assert not torch.allclose(out, out_perm, atol=1e-5)
