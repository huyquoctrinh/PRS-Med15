from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import json

import numpy as np

import pytest
import torch
from torch.utils.data import DataLoader
from transformers import (
    SamConfig,
    SamMaskDecoderConfig,
    SamModel,
    SamPromptEncoderConfig,
    SamVisionConfig,
)

from prs_med15.data.collate import collate_dialogues
from prs_med15.data.dataset import DatasetConfig, DialogueDataset
from prs_med15.data.schema import Action, Dialogue, ObjectRecord, Scan, Turn
from prs_med15.models.factory import build_model
from prs_med15.models.mask_decoder import MedSAMMaskDecoder
from prs_med15.cli import main as cli_main
from prs_med15.training.trainer import fit


def tiny_sam() -> SamModel:
    vision = SamVisionConfig(
        hidden_size=32,
        output_channels=16,
        num_hidden_layers=1,
        num_attention_heads=4,
        image_size=32,
        patch_size=8,
        num_pos_feats=8,
        global_attn_indexes=[0],
        window_size=4,
    )
    prompt = SamPromptEncoderConfig(
        hidden_size=16,
        image_size=32,
        patch_size=8,
        mask_input_channels=8,
    )
    mask = SamMaskDecoderConfig(
        hidden_size=16,
        mlp_dim=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        attention_downsample_rate=2,
        iou_head_hidden_dim=16,
    )
    return SamModel(
        SamConfig(
            vision_config=vision,
            prompt_encoder_config=prompt,
            mask_decoder_config=mask,
        )
    )


@pytest.mark.parametrize("with_prior", [False, True])
def test_hf_sam_decoder_runtime_grid_and_mask_prompt(with_prior):
    torch.manual_seed(3)
    sam = tiny_sam()
    decoder = MedSAMMaskDecoder(
        sam.prompt_encoder,
        sam.mask_decoder,
        sam.shared_image_embedding,
        hidden_dim=12,
        sam_dim=16,
        microbatch=2,
    )
    features = torch.randn(3, 16, 4, 4)
    prior = torch.rand(3, 32, 32) if with_prior else None
    logits = decoder(features, torch.randn(12), (32, 32), prior_mask=prior)
    assert logits.shape == (3, 1, 32, 32)
    expected = sam.shared_image_embedding(
        torch.stack(
            torch.meshgrid(
                (torch.arange(4) + 0.5) / 4,
                (torch.arange(4) + 0.5) / 4,
                indexing="ij",
            )[::-1],
            dim=-1,
        )
    ).permute(2, 0, 1).unsqueeze(0)
    assert torch.allclose(decoder.image_positional_embeddings(4, 4, torch.device("cpu"), torch.float32), expected)
    dense = decoder._dense_prompt(prior, 3, 4, 4)
    assert dense.shape == (3, 16, 4, 4)
    if prior is None:
        assert torch.equal(dense[0, :, 0, 0], sam.prompt_encoder.no_mask_embed.weight[0])


def make_dialogue(depth: int = 1) -> tuple[Dialogue, callable, callable]:
    first = "<CREATE> <SCAN_REF> S1 <ENTITY> E1 [SEG] </CREATE>"
    second = "<CREATE> <SCAN_REF> S1 <ENTITY> E2 <REF> S1:E1 [SEG] </CREATE>"
    dialogue = Dialogue(
        "tiny-dialogue",
        "patient",
        "case",
        "synthetic",
        "train",
        [Scan("S1", "unused", None, "CT", "native_2d" if depth == 1 else "volume")],
        {
            "E1": ObjectRecord("o1", "organ", mask_path="unused"),
            "E2": ObjectRecord("o2", "lesion", mask_path="unused"),
        },
        [
            Turn("T1", "segment organ", first, None, None, "S1", [Action("create", "S1", "E1", "o1")]),
            Turn(
                "T2",
                "segment lesion in it",
                second,
                None,
                None,
                "S1",
                [Action("create", "S1", "E2", "o2", refs=["S1:E1"])],
                ["E1"],
            ),
        ],
    )
    image_loader = lambda _: (
        torch.rand(3, depth, 8, 8),
        torch.ones(depth, 8, 8, dtype=torch.bool),
        {"unknown_geometry": True},
    )
    mask_loader = lambda _dialogue, entity: (
        torch.ones(depth, 8, 8, dtype=torch.bool)
        if entity == "E1"
        else torch.eye(8, dtype=torch.bool).expand(depth, -1, -1)
    )
    return dialogue, image_loader, mask_loader


def tiny_cfg(results: Path, max_steps: int, resume: Path | None = None):
    train = {
        "stage": "a",
        "history_mode": "oracle",
        "batch_size": 1,
        "learning_rate": 1e-3,
        "encoder_learning_rate": 0.0,
        "max_steps": max_steps,
        "gradient_accumulation": 1,
        "warmup_steps": 0,
        "checkpoint_every": 1,
        "validate_every": 1,
        "results_dir": str(results),
        "loss": {"text": 1.0, "mask": 1.0},
    }
    if resume is not None:
        train["resume"] = str(resume)
    return SimpleNamespace(
        train=train,
        model={"tiny": {"max_len": 256}},
    )


def make_loader(model, tokenizer):
    dialogue, image_loader, mask_loader = make_dialogue()
    config = DatasetConfig(
        scan_loader=image_loader,
        mask_loader=mask_loader,
        image_size=8,
        d_max=1,
        max_len=256,
        global_budget=model.global_pooler.budget,
        detail_tokens=model.query_encoder.slots.shape[0],
        region_tokens_per_entity=model.region_tokenizer.tokens_per_entity,
    )
    dataset = DialogueDataset([dialogue], tokenizer, config)
    return DataLoader(
        dataset,
        batch_size=1,
        collate_fn=lambda rows: collate_dialogues(rows, tokenizer.pad_token_id),
    )


def test_memory_records_are_exact_and_oracle_unrolling_is_equivalent(tmp_path):
    torch.manual_seed(11)
    cfg = tiny_cfg(tmp_path / "unused", 1)
    oracle, tokenizer, _ = build_model(cfg, tiny=True)
    forced = deepcopy(oracle)
    forced.history_mode = "mixed"
    forced.oracle_fraction = 1.0
    batch = next(iter(make_loader(oracle, tokenizer)))
    assert all(len(record["positions"]) == 4 for record in batch["memory_records"])
    oracle.eval()
    forced.eval()
    first = oracle(batch)
    second = forced(batch)
    assert torch.allclose(first.logits, second.logits)
    assert all(torch.allclose(left, right) for left, right in zip(first.mask_logits, second.mask_logits))
    broken = deepcopy(batch)
    broken["items"][0]["memory_records"][0]["positions"].pop()
    with pytest.raises(ValueError, match="slots"):
        oracle(broken)


def test_fit_two_steps_then_resume_cpu(tmp_path):
    first_dir = tmp_path / "first"
    torch.manual_seed(17)
    cfg = tiny_cfg(first_dir, 2)
    model, tokenizer, _ = build_model(cfg, tiny=True)
    loader = make_loader(model, tokenizer)
    result = fit(model, loader, loader, cfg)
    assert result["step"] == 2
    checkpoint = first_dir / "checkpoint"
    expected = {
        "lora_adapter",
        "vocab_deltas.pt",
        "new_modules.pt",
        "optimizer.pt",
        "trainer_state.json",
    }
    assert expected <= {path.name for path in checkpoint.iterdir()}

    torch.manual_seed(17)
    resumed_cfg = tiny_cfg(tmp_path / "resumed", 3, checkpoint)
    resumed, resumed_tokenizer, _ = build_model(resumed_cfg, tiny=True)
    resumed_loader = make_loader(resumed, resumed_tokenizer)
    continued = fit(resumed, resumed_loader, None, resumed_cfg)
    assert continued["step"] == 3


def test_cli_synthetic_prepare_and_tiny_train(tmp_path):
    dialogues = tmp_path / "dialogues" / "amos_3d"
    dialogues.mkdir(parents=True)
    manifests = tmp_path / "manifests"
    image = tmp_path / "image.npz"
    labels = tmp_path / "labels"
    labels.mkdir()
    np.savez(
        image,
        image=np.random.default_rng(1).random((3, 1, 8, 8), dtype=np.float32),
        valid_mask=np.ones((1, 8, 8), dtype=bool),
    )
    np.save(labels / "mask.npy", np.ones((1, 8, 8), dtype=bool))
    source_path = dialogues / "dialogues_train.jsonl"
    with source_path.open("w") as stream:
        for index in range(2):
            record = {
                "id": f"dialogue-{index}",
                "dataset": "amos22",
                "case": f"case-{index}",
                "split": "train",
                "modality": "CT",
                "image": str(image),
                "label": str(labels),
                "objects": {
                    "o1": {"canonical": "organ", "mask_file": "mask.npy"},
                },
                "conversations": [
                    {"from": "human", "value": "segment organ"},
                    {"from": "gpt", "value": "Organ [SEG]", "seg_objects": ["o1"]},
                ],
            }
            stream.write(json.dumps(record) + "\n")
    sources_config = tmp_path / "sources.yaml"
    sources_config.write_text(
        "sources:\n"
        f"  dialogues_root: {tmp_path / 'dialogues'}\n"
        f"  mrmedseg_root: {tmp_path / 'missing_mr'}\n"
        f"  manifest_root: {manifests}\n"
    )
    assert cli_main([
        "prepare",
        "--config",
        str(sources_config),
        "--sources",
        "amos",
        "--limit",
        "2",
    ]) == 0
    train_config = tmp_path / "train.yaml"
    results = tmp_path / "results"
    train_config.write_text(
        "runtime: {max_len: 256, num_workers: 0, seed: 1}\n"
        f"data: {{manifest: {manifests / 'train.jsonl'}, image_size: 8, d_max: 1}}\n"
        "model: {tiny: {max_len: 256}}\n"
        "train:\n"
        "  stage: a\n"
        "  batch_size: 1\n"
        "  learning_rate: 0.001\n"
        "  encoder_learning_rate: 0.0\n"
        "  max_steps: 2\n"
        "  checkpoint_every: 2\n"
        "  loss: {text: 1.0, mask: 1.0}\n"
    )
    assert cli_main([
        "train",
        "--config",
        str(train_config),
        "--tiny",
        "--limit",
        "2",
        "--results_dir",
        str(results),
    ]) == 0
    state = json.loads((results / "checkpoint" / "trainer_state.json").read_text())
    assert state["step"] == 2


def test_skipping_text_pass_keeps_mask_path_identical(tmp_path):
    """Stage A (text weight 0) skips the full-turn LLM pass; masks and mask grads must not change."""
    from prs_med15.training.losses import total_loss

    torch.manual_seed(5)
    cfg = tiny_cfg(tmp_path / "unused", 1)
    full, tokenizer, _ = build_model(cfg, tiny=True)
    skipped = deepcopy(full)
    skipped.compute_text_logits = False
    batch = next(iter(make_loader(full, tokenizer)))
    full.eval()
    skipped.eval()
    left = full(batch)
    right = skipped(batch)
    assert right.logits.shape[-1] == 1
    assert all(torch.allclose(a, b) for a, b in zip(left.mask_logits, right.mask_logits))
    loss_left, _ = total_loss(left.logits, left.labels, left.actions, lambda_text=0.0)
    loss_right, _ = total_loss(right.logits, right.labels, right.actions, lambda_text=0.0)
    loss_left.backward()
    loss_right.backward()
    assert torch.allclose(loss_left, loss_right)
    for (name, a), (_, b) in zip(full.named_parameters(), skipped.named_parameters()):
        if a.grad is not None and b.grad is not None:
            assert torch.allclose(a.grad, b.grad, atol=1e-6), name


def test_factory_disables_text_pass_when_text_weight_zero(tmp_path):
    cfg = tiny_cfg(tmp_path / "unused", 1)
    cfg.train["loss"] = {"text": 0.0, "mask": 1.0}
    model, _, _ = build_model(cfg, tiny=True)
    assert model.compute_text_logits is False


def test_global_tokens_match_reserved_slots_for_letterboxed_slices():
    """Narrow letterboxed slices (empty 4x4 cells) must still emit 16 tokens per valid slice."""
    from prs_med15.models.global_tokens import GlobalTokenPooler, global_token_count

    pooler = GlobalTokenPooler(feature_dim=8, d_llm=16, budget=512)
    dense = torch.randn(1, 5, 32, 32, 8)
    valid = torch.zeros(1, 5, 32, 32, dtype=torch.bool)
    valid[:, :4, 14:18, :] = True  # 4 valid slices, a thin horizontal band only
    out = pooler(dense, valid)
    assert int(out.token_mask.sum()) == global_token_count(4, 512) == 64
    deep = torch.zeros(1, 40, 32, 32, dtype=torch.bool)
    deep[:, :, 14:18, :] = True
    out = pooler(torch.randn(1, 40, 32, 32, 8), deep)
    assert int(out.token_mask.sum()) == global_token_count(40, 512)
