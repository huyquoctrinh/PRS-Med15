"""Feature cache, rank-matched sampling."""
from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import torch

from prs_med15.data.batching import SourceBalancedBucketBatchSampler
from prs_med15.data.collate import collate_dialogues
from prs_med15.data.dataset import DatasetConfig, DialogueDataset
from prs_med15.data.features import feature_path, load_feature_rows, save_features
from prs_med15.data.schema import Action, Turn
from prs_med15.models.factory import build_model
from prs_med15.training.losses import total_loss

from .test_phase4 import _dialogue


def _rows():
    rows = []
    for index in range(40):
        row = _dialogue("unused", "unused", f"d{index}")
        if index % 3 == 0:
            row.scans[0] = replace(row.scans[0], input_kind="native_2d")
        if index % 4 == 0:  # heavier dialogue
            row.turns.append(Turn("T2", "again", "ok", None, None, "S1",
                                  [Action("create", "S1", "E1", "organ")]))
        rows.append(row)
    return rows


def test_rank_matched_sampler_pairs_same_bucket_and_stays_disjoint():
    rows = _rows()
    world = 2
    samplers = [SourceBalancedBucketBatchSampler(rows, 1, seed=3, rank=r, world_size=world,
                                                 match_ranks=True, match_window=4)
                for r in range(world)]
    per_rank = [list(s) for s in samplers]
    assert len(per_rank[0]) == len(per_rank[1]) > 0
    for a, b in zip(*per_rank):
        assert samplers[0].bucket_of[a[0]] == samplers[0].bucket_of[b[0]]
    flat = [i for batches in per_rank for batch in batches for i in batch]
    assert len(flat) == len(set(flat))
    base = SourceBalancedBucketBatchSampler(rows, 1, seed=3, rank=0, world_size=1)
    assert set(flat) <= {i for batch in base for i in batch}
    # Without matching, the same seed gives the previous order (back-compatible).
    old = SourceBalancedBucketBatchSampler(rows, 1, seed=3, rank=0, world_size=2)
    assert list(old) == list(SourceBalancedBucketBatchSampler(rows, 1, seed=3, rank=0, world_size=2))


def test_feature_rows_are_slab_slices(tmp_path):
    full = np.random.default_rng(0).standard_normal((10, 2, 2, 3)).astype(np.float16)
    path = feature_path(tmp_path, "src", "img.nii.gz", {"cache_path": "/x/abc123.npz"})
    assert path.name == "abc123.npy"
    save_features(path, full)
    rows = load_feature_rows(path, 3, 7, expected_depth=10)
    assert torch.equal(rows, torch.from_numpy(full[3:7]))
    assert load_feature_rows(path, 0, 2, expected_depth=11) is None  # stale/mismatched file
    assert load_feature_rows(tmp_path / "missing.npy", 0, 1, 1) is None


def test_cached_features_match_online_encoder(tmp_path):
    torch.manual_seed(0)
    cfg = SimpleNamespace(train={"stage": "a", "loss": {"text": 0.0, "mask": 1.0}},
                          model={"tiny": {"max_len": 256}})
    model, tokenizer, _ = build_model(cfg, tiny=True)
    model.eval()
    image = torch.rand(3, 1, 8, 8)
    valid = torch.ones(1, 8, 8, dtype=torch.bool)
    valid[:, :, 6:] = False
    common = dict(image_size=8, d_max=1, max_len=256, global_budget=model.global_pooler.budget,
                  detail_tokens=model.query_encoder.slots.shape[0],
                  scan_loader=lambda _s: (image, valid, {"unknown_geometry": True}),
                  mask_loader=lambda _d, _e: torch.ones(1, 8, 8, dtype=torch.bool))
    row = _dialogue("img.png", "unused", "d0")
    with torch.no_grad():
        feats = model.encoder(image[None], valid[None]).dense[0].numpy()
    save_features(feature_path(tmp_path, row.source, row.scans[0].image_path, {}), feats)
    online = DialogueDataset([row], tokenizer, DatasetConfig(**common))[0]
    cached = DialogueDataset([row], tokenizer, DatasetConfig(**common, feature_root=str(tmp_path)))[0]
    assert online["encoder_features"] is None and cached["encoder_features"] is not None
    losses = []
    for item in (online, cached):
        batch = collate_dialogues([item])
        with torch.no_grad():
            out = model(batch)
            losses.append(total_loss(out.logits.float(), out.labels, out.actions,
                                     lambda_text=0.0, lambda_mask=1.0)[0])
    assert torch.allclose(losses[0], losses[1], atol=1e-3)


def test_stage_a_checkpoint_initialises_lora_stage_b(tmp_path):
    from prs_med15.training.trainer import load_component_checkpoint, save_training_checkpoint

    def make(stage):
        torch.manual_seed(0 if stage == "a" else 1)
        cfg = SimpleNamespace(train={"stage": stage, "loss": {"text": 1.0, "mask": 1.0}},
                              model={"tiny": {"max_len": 256}})
        return build_model(cfg, tiny=True)[0]

    a, b = make("a"), make("b")
    with torch.no_grad():
        for p in a.parameters():
            if p.requires_grad:
                p.add_(torch.randn_like(p))
    opt = torch.optim.AdamW([p for p in a.parameters() if p.requires_grad])
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda _: 1.0)
    save_training_checkpoint(tmp_path / "ck", a, opt, sched, 1, 0, 0)
    load_component_checkpoint(tmp_path / "ck", b)
    a_state, b_state = a.state_dict(), b.state_dict()
    checked = 0
    for name, p in a.named_parameters():
        if not p.requires_grad:
            continue
        key = name if name in b_state else name.replace("language_model.", "language_model.base_model.model.", 1)
        assert torch.equal(a_state[name], b_state[key]), name
        checked += 1
    assert checked > 5


def _three_turn_dialogue():
    from prs_med15.data.schema import Dialogue, ObjectRecord, Scan
    create = "<CREATE> <SCAN_REF> S1 <ENTITY> {e} [SEG] </CREATE>"
    return Dialogue(
        "d3", "p3", "c3", "synthetic", "train",
        [Scan("S1", "img.png", "lbl", "MR", "volume")],
        {"E1": ObjectRecord("o1", "liver", label_id=1), "E2": ObjectRecord("o2", "kidney", label_id=2)},
        [
            Turn("T1", "segment the liver", "<think>liver " + create.format(e="E1") + "</think> done <REF> S1:E1",
                 None, None, "S1", [Action("create", "S1", "E1", "o1")]),
            Turn("T2", "is it large?", "The liver <REF> S1:E1 is large.", None, None, "S1", [],
                 reference_entity_ids=["S1:E1"]),
            Turn("T3", "now the kidney next to it", "kidney " + create.format(e="E2"), None, None, "S1",
                 [Action("create", "S1", "E2", "o2", refs=["S1:E1"])], reference_entity_ids=["S1:E1"]),
        ],
    )


def test_fused_turn_pass_matches_two_pass_loss_and_gradients():
    cfg = SimpleNamespace(train={"stage": "b", "loss": {"text": 1.0, "mask": 1.0}},
                          model={"tiny": {"max_len": 512}})
    torch.manual_seed(0)
    model, tokenizer, _ = build_model(cfg, tiny=True)
    model.eval()  # deterministic (no dropout); gradients still flow
    masks = {"E1": torch.zeros(2, 8, 8, dtype=torch.bool), "E2": torch.zeros(2, 8, 8, dtype=torch.bool)}
    masks["E1"][:, 1:4, 1:5] = True
    masks["E2"][:, 4:7, 2:6] = True
    image = torch.rand(3, 2, 8, 8)
    dataset = DialogueDataset([_three_turn_dialogue()], tokenizer, DatasetConfig(
        image_size=8, d_max=2, max_len=512, global_budget=model.global_pooler.budget,
        detail_tokens=model.query_encoder.slots.shape[0],
        scan_loader=lambda _s: (image, torch.ones(2, 8, 8, dtype=torch.bool), {"unknown_geometry": True}),
        mask_loader=lambda _d, entity: masks[entity]))
    batch = collate_dialogues([dataset[0]])
    assert len(batch["items"][0]["turn_boundaries"]) == 3 and len(batch["actions"]) == 2

    results = []
    for fused in (False, True):
        for text_weight in (1.0, 0.0):
            model.fused_turn_pass = fused
            model.compute_text_logits = text_weight > 0
            model.zero_grad(set_to_none=True)
            out = model(batch)
            loss, _ = total_loss(out.logits.float(), out.labels, out.actions,
                                 lambda_text=text_weight, lambda_mask=1.0)
            loss.backward()
            grads = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}
            results.append((fused, text_weight, loss.detach(), grads))
    for text_weight in (1.0, 0.0):
        (_, _, l_old, g_old), = [r for r in results if not r[0] and r[1] == text_weight]
        (_, _, l_new, g_new), = [r for r in results if r[0] and r[1] == text_weight]
        assert torch.allclose(l_old, l_new, atol=1e-5), (text_weight, l_old, l_new)
        assert g_old.keys() == g_new.keys() and len(g_old) > 10
        for name in g_old:
            assert torch.allclose(g_old[name], g_new[name], atol=1e-5, rtol=1e-4), name


def test_sampler_start_batch_skips_once():
    rows = _rows()
    sampler = SourceBalancedBucketBatchSampler(rows, 1, seed=3, rank=0, world_size=2, match_ranks=True)
    full = list(sampler)
    sampler.start_batch = 5
    assert list(sampler) == full[5:]
    assert list(sampler) == full  # one-shot
