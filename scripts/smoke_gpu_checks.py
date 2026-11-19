"""GPU-only checks invoked by smoke_gpu.sh; never imported by the package."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch

from prs_med15.config import load_config
from prs_med15.data.collate import collate_dialogues
from prs_med15.data.dataset import DatasetConfig, DialogueDataset
from prs_med15.data.schema import Dialogue
from prs_med15.models.factory import build_model
from prs_med15.models.llm_bridge import load_language_model


ROOT = Path(__file__).resolve().parents[1]
LLM_PATH = Path("/data2/fcq0219/models/llava_med")
OLD_REPO = Path("/data2/gdp2988/prs_med/PRS-Med")


def rows(path: Path) -> list[Dialogue]:
    with path.open() as stream:
        return [Dialogue.from_json(line) for line in stream if line.strip()]


def move(value, device):
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, dict):
        return {key: move(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [move(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(move(item, device) for item in value)
    return value


@torch.no_grad()
def parity() -> None:
    sys.path.insert(0, str(OLD_REPO))
    from llava.model.language_model.llava_mistral import LlavaMistralForCausalLM
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(LLM_PATH, local_files_only=True)
    ids = tokenizer("Describe the scan briefly.", return_tensors="pt")["input_ids"].cuda()
    fork = LlavaMistralForCausalLM.from_pretrained(
        LLM_PATH,
        local_files_only=True,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
    ).cuda().eval()
    fork_logits = fork(input_ids=ids).logits.float().cpu()
    del fork
    torch.cuda.empty_cache()
    direct, _ = load_language_model(LLM_PATH, dtype=torch.float32)
    direct = direct.cuda().eval()
    direct_logits = direct(input_ids=ids).logits.float().cpu()
    difference = float((fork_logits - direct_logits).abs().max())
    print(json.dumps({"check": "text_logit_parity", "max_abs_difference": difference}))
    if difference > 2e-3:
        raise RuntimeError(f"text-only parity failed: {difference}")


def production():
    cfg = load_config(ROOT / "configs/train/stage_a.yaml")
    cfg.model = load_config(ROOT / "configs/model/shared_slice.yaml").model
    return cfg, build_model(cfg, tiny=False)


@torch.no_grad()
def run_dialogues(selected: list[Dialogue], label: str) -> None:
    cfg, (model, tokenizer, _) = production()
    model = model.cuda().eval()
    dataset = DialogueDataset(
        selected,
        tokenizer,
        DatasetConfig(
            max_len=cfg.runtime.max_len,
            image_size=cfg.data.image_size,
            d_max=cfg.data.d_max,
            global_budget=cfg.model["global"]["budget"],
            detail_tokens=cfg.model["detail"]["K"],
        ),
        mode="eval",
    )
    torch.cuda.reset_peak_memory_stats()
    for item in dataset:
        batch = move(collate_dialogues([item], tokenizer.pad_token_id or 0), torch.device("cuda"))
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            output = model(batch)
        if not output.mask_logits:
            raise RuntimeError(f"{label} example produced no masks")
    print(json.dumps({
        "check": label,
        "examples": len(selected),
        "peak_memory_bytes": torch.cuda.max_memory_allocated(),
    }))


def forward() -> None:
    two_d = rows(ROOT / "data/manifests/mrmedseg_train.jsonl")[:1]
    three_d = rows(ROOT / "data/manifests/amos_train.jsonl")[:1]
    run_dialogues(two_d + three_d, "2d_3d_forward")


def dialogue() -> None:
    candidates = rows(ROOT / "data/manifests/mrmedseg_train.jsonl")
    selected = next(
        row
        for row in candidates
        if len(row.turns) >= 2
        and any(turn.reference_entity_ids for turn in row.turns[1:])
        and sum(len(turn.actions) for turn in row.turns) >= 2
    )
    run_dialogues([selected], "create_ref_referenced_create")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("check", choices=("parity", "forward", "dialogue"))
    args = parser.parse_args()
    globals()[args.check]()


if __name__ == "__main__":
    main()
