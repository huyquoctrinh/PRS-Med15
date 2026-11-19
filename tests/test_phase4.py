from __future__ import annotations

from functools import partial
import os
from pathlib import Path
import socket
from types import SimpleNamespace

import nibabel as nib
import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader

from prs_med15.data.batching import SourceBalancedBucketBatchSampler
from prs_med15.data.cache import build_cache
from prs_med15.data.collate import collate_dialogues
from prs_med15.data.dataset import DatasetConfig, DialogueDataset
from prs_med15.data.preprocess import PreprocessConfig
from prs_med15.data.schema import Action, Dialogue, ObjectRecord, Scan, Turn
from prs_med15.models.factory import TinyTokenizer, build_model
from prs_med15.training.trainer import fit


def _dialogue(image: str, label: str, dialogue_id: str = "d0") -> Dialogue:
    return Dialogue(
        dialogue_id,
        f"patient-{dialogue_id}",
        f"case-{dialogue_id}",
        "synthetic",
        "train",
        [Scan("S1", image, label, "MR", "volume")],
        {"E1": ObjectRecord("organ", "organ", label_id=1)},
        [
            Turn(
                "T1",
                "segment organ",
                "<CREATE> <SCAN_REF> S1 <ENTITY> E1 [SEG] </CREATE>",
                None,
                None,
                "S1",
                [Action("create", "S1", "E1", "organ")],
            )
        ],
    )


def test_cached_and_raw_dialogue_tensors_match(tmp_path):
    image_path = tmp_path / "image.nii.gz"
    label_path = tmp_path / "label.nii.gz"
    xyz = np.linspace(0, 1, 5 * 4 * 3, dtype=np.float32).reshape(5, 4, 3)
    label = np.zeros((5, 4, 3), dtype=np.int16)
    label[1:4, 1:3, 1:] = 1
    nib.save(nib.Nifti1Image(xyz, np.diag([1.2, 1.5, 2.0, 1.0])), image_path)
    nib.save(nib.Nifti1Image(label, np.diag([1.2, 1.5, 2.0, 1.0])), label_path)
    dialogue = _dialogue(str(image_path), str(label_path))
    manifest = tmp_path / "synthetic_train.jsonl"
    manifest.write_text(dialogue.to_json() + "\n")
    preprocess_config = PreprocessConfig(image_size=12, d_max=8)
    report = build_cache(
        [manifest],
        tmp_path / "processed",
        preprocess_config,
        workers=1,
    )
    assert report["built"] == 1 and report["failed"] == 0

    tokenizer = TinyTokenizer()
    common = {
        "image_size": 12,
        "d_max": 8,
        "max_len": 256,
        "global_budget": 16,
        "detail_tokens": 4,
        "processed_root": str(tmp_path / "processed"),
    }
    cached = DialogueDataset([dialogue], tokenizer, DatasetConfig(**common, use_cache=True))[0]
    raw = DialogueDataset([dialogue], tokenizer, DatasetConfig(**common, use_cache=False))[0]
    assert torch.allclose(cached["scans"][0], raw["scans"][0], atol=5e-4, rtol=0)
    assert torch.equal(cached["valid_masks"][0], raw["valid_masks"][0])
    assert cached["geometry"][0]["transform"] == raw["geometry"][0]["transform"]
    assert torch.equal(cached["actions"][0]["target"], raw["actions"][0]["target"])


def test_distributed_bucket_sampler_is_disjoint_and_epoch_seeded():
    rows = [_dialogue("unused", "unused", f"d{index}") for index in range(8)]
    left = SourceBalancedBucketBatchSampler(rows, 1, seed=4, rank=0, world_size=2)
    right = SourceBalancedBucketBatchSampler(rows, 1, seed=4, rank=1, world_size=2)
    left_indices = {index for batch in left for index in batch}
    right_indices = {index for batch in right for index in batch}
    assert left_indices.isdisjoint(right_indices)
    assert len(left) == len(right) == 4
    first_epoch = list(left)
    left.set_epoch(1)
    assert list(left) != first_epoch


def _ddp_worker(rank: int, world_size: int, port: int, output: str) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    torch.manual_seed(9)
    train = {
        "stage": "a",
        "batch_size": 1,
        "learning_rate": 1e-3,
        "encoder_learning_rate": 0.0,
        "max_steps": 2,
        "gradient_accumulation": 2,
        "warmup_steps": 0,
        "checkpoint_every": 2,
        "validate_every": 2,
        "results_dir": str(Path(output) / "results"),
        "checkpoint_dir": str(Path(output) / "checkpoints"),
        "loss": {"text": 1.0, "mask": 1.0},
    }
    cfg = SimpleNamespace(train=train, model={"tiny": {"max_len": 256}})
    model, tokenizer, _ = build_model(cfg, tiny=True)
    model = DistributedDataParallel(
        model,
        find_unused_parameters=True,
        static_graph=False,
    )
    rows = [_dialogue("unused", "unused", f"d{index}") for index in range(8)]
    dataset = DialogueDataset(
        rows,
        tokenizer,
        DatasetConfig(
            image_size=8,
            d_max=1,
            max_len=256,
            global_budget=model.module.global_pooler.budget,
            detail_tokens=model.module.query_encoder.slots.shape[0],
            scan_loader=lambda _scan: (
                torch.ones(3, 1, 8, 8),
                torch.ones(1, 8, 8, dtype=torch.bool),
                {"unknown_geometry": True},
            ),
            mask_loader=lambda _dialogue, _entity: torch.ones(1, 8, 8, dtype=torch.bool),
        ),
    )
    sampler = SourceBalancedBucketBatchSampler(
        rows,
        1,
        seed=7,
        rank=rank,
        world_size=world_size,
    )
    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        collate_fn=partial(collate_dialogues, pad_token_id=tokenizer.pad_token_id),
    )
    result = fit(model, loader, None, cfg)
    assert result["step"] == 2
    dist.destroy_process_group()


def test_two_process_gloo_tiny_training_writes_one_checkpoint(tmp_path):
    try:
        with socket.socket() as server:
            server.bind(("127.0.0.1", 0))
            port = server.getsockname()[1]
    except PermissionError:
        pytest.skip("sandbox forbids localhost sockets required by torch.distributed gloo")
    mp.spawn(_ddp_worker, args=(2, port, str(tmp_path)), nprocs=2, join=True)
    checkpoints = list((tmp_path / "checkpoints").glob("checkpoint-*"))
    assert [path.name for path in checkpoints] == ["checkpoint-2"]
    assert (checkpoints[0] / "trainer_state.json").is_file()


def _grad_avg_worker(rank, world, port, out_dir):
    import os
    import torch
    import torch.distributed as dist
    from prs_med15.training.trainer import average_gradients, broadcast_trainable_parameters

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group("gloo", rank=rank, world_size=world)
    torch.manual_seed(100 + rank)  # different init per rank -> broadcast must align them
    model = torch.nn.ModuleDict({"a": torch.nn.Linear(3, 2), "b": torch.nn.Linear(3, 2)})
    broadcast_trainable_parameters(model)
    x = torch.ones(1, 3) * (rank + 1)
    loss = model["a"](x).sum()
    if rank == 0:  # module "b" unused on rank 1 -> its grad must count as zero there
        loss = loss + model["b"](x).sum()
    loss.backward()
    average_gradients(model, bucket_elements=4)
    torch.save({k: v.grad for k, v in model.named_parameters()}, f"{out_dir}/grads_{rank}.pt")
    torch.save({k: v.detach() for k, v in model.named_parameters()}, f"{out_dir}/weights_{rank}.pt")
    dist.destroy_process_group()


def test_manual_gradient_averaging_with_rank_dependent_unused_modules(tmp_path):
    import socket
    import torch
    import torch.multiprocessing as mp

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    mp.spawn(_grad_avg_worker, args=(2, port, str(tmp_path)), nprocs=2, join=True)
    g0, g1 = (torch.load(tmp_path / f"grads_{r}.pt") for r in (0, 1))
    w0, w1 = (torch.load(tmp_path / f"weights_{r}.pt") for r in (0, 1))
    for key in g0:
        assert torch.allclose(g0[key], g1[key]), key
        assert torch.allclose(w0[key], w1[key]), key
    # a.weight grad: mean over ranks of x = (1+2)/2 = 1.5 per input, for each of 2 outputs
    assert torch.allclose(g0["a.weight"], torch.full((2, 3), 1.5))
    # b.weight grad: rank0 contributes x=1, rank1 contributes 0 -> 0.5
    assert torch.allclose(g0["b.weight"], torch.full((2, 3), 0.5))
