import torch

from prs_med15.training.losses import dialogue_mask_loss, hard_negative_loss


def _action(entity, target, logits, batch_idx=0, scan_idx=0):
    return {"entity_id": entity, "target": target, "logits": logits, "batch_idx": batch_idx,
            "scan_idx": scan_idx, "valid": torch.ones_like(target)}


def test_hard_negative_penalises_other_entity_voxels_only():
    a = torch.zeros(2, 8, 8, dtype=torch.bool); a[:, :4] = True
    b = torch.zeros(2, 8, 8, dtype=torch.bool); b[:, 2:6] = True   # overlaps a on rows 2-3
    logits = torch.full((2, 1, 8, 8), -10.0); logits[:, :, 4:6] = 10.0  # predicts b-only rows
    actions = [_action("E1", a, logits), _action("E2", b, torch.zeros(2, 1, 8, 8))]
    neg = hard_negative_loss(actions)
    assert neg[0] is not None and abs(float(neg[0]) - 1.0) < 1e-3   # b \\ a = rows 4-5, all predicted
    assert abs(float(neg[1]) - 0.5) < 1e-3                          # a \\ b, sigmoid(0)=0.5


def test_hard_negative_ignores_other_scans_and_same_entity():
    a = torch.ones(1, 4, 4, dtype=torch.bool)
    actions = [_action("E1", a, torch.zeros(1, 1, 4, 4)),
               _action("E2", ~a | a, torch.zeros(1, 1, 4, 4), scan_idx=1),
               _action("E1", a, torch.zeros(1, 1, 4, 4))]
    assert hard_negative_loss(actions)[0] is None


def test_negative_weight_zero_matches_old_loss():
    a = torch.zeros(1, 8, 8, dtype=torch.bool); a[:, :3] = True
    b = torch.zeros(1, 8, 8, dtype=torch.bool); b[:, 5:] = True
    actions = [_action("E1", a, torch.randn(1, 1, 8, 8)), _action("E2", b, torch.randn(1, 1, 8, 8))]
    ref = torch.zeros(1)
    base = dialogue_mask_loss(actions, ref)
    assert torch.allclose(base, dialogue_mask_loss(actions, ref, lambda_negative=0.0))
    assert dialogue_mask_loss(actions, ref, lambda_negative=1.0) > base


def test_medsam_decoder_film_and_extra_tokens_shapes():
    from prs_med15.models.mask_decoder import MedSAMMaskDecoder
    try:
        from transformers import SamConfig, SamModel
    except ImportError:  # pragma: no cover
        return
    cfg = SamConfig()
    cfg.mask_decoder_config.num_hidden_layers = 1
    cfg.vision_config.num_hidden_layers = 1
    cfg.vision_config.hidden_size = 64
    cfg.vision_config.num_attention_heads = 2
    sam = SamModel(cfg)
    torch.manual_seed(0)
    base = MedSAMMaskDecoder(sam.prompt_encoder, sam.mask_decoder, sam.shared_image_embedding,
                             hidden_dim=48, sam_dim=256)
    new = MedSAMMaskDecoder(sam.prompt_encoder, sam.mask_decoder, sam.shared_image_embedding,
                            hidden_dim=48, sam_dim=256, film=True)
    new.query.load_state_dict(base.query.state_dict())
    feats, state = torch.randn(3, 256, 8, 8), torch.randn(48)
    # zero-init FiLM: identical function at initialisation
    assert torch.allclose(base(feats, state, (16, 16)), new(feats, state, (16, 16)), atol=1e-5)
    extra = MedSAMMaskDecoder(sam.prompt_encoder, sam.mask_decoder, sam.shared_image_embedding,
                              hidden_dim=48, sam_dim=256, extra_tokens=6, film=True)
    out = extra(feats, state, (16, 16))
    assert out.shape == (3, 1, 16, 16)
    out.sum().backward()
    assert extra.query_extra[0].weight.grad.abs().sum() > 0
    assert extra.film[-1].weight.grad.abs().sum() > 0
