from pathlib import Path

import numpy as np
from PIL import Image
import torch

from prs_med15.data.dataset import DatasetConfig, DialogueDataset
from prs_med15.data.schema import Action, Dialogue, ObjectRecord, Scan, Turn
from tests.fake_tokenizer import FakeTokenizer


def test_assistant_labels_seg_positions_and_placeholders(tmp_path: Path):
    mask_path = tmp_path / "mask.png"
    Image.fromarray(np.array([[0, 255], [0, 0]], np.uint8)).save(mask_path)
    dialogue = Dialogue(
        "d", "amos:c", "c", "amos", "train",
        [Scan("S1", "unused", None, "CT", "volume")],
        {"E1": ObjectRecord("o1", "kidney", mask_path=str(mask_path))},
        [Turn("T1", "segment", "reason <CREATE> <SCAN_REF> S1 <ENTITY> E1 [SEG] </CREATE>",
              "ground", "create", "S1", [Action("create", "S1", "E1", "o1")])],
    )
    def scan_loader(_):
        return torch.ones(3, 2, 8, 8), torch.ones(2, 8, 8, dtype=torch.bool), {}
    tokenizer = FakeTokenizer()
    item = DialogueDataset([dialogue], tokenizer, DatasetConfig(scan_loader=scan_loader))[0]
    assert len(item["actions"]) == 1
    pos = item["actions"][0]["seg_token_pos"]
    assert item["input_ids"][pos].item() == tokenizer.vocab["[SEG]"]
    assert item["labels"][pos].item() == tokenizer.vocab["[SEG]"]
    assert item["actions"][0]["target"].shape == (2, 8, 8)
    for positions in item["slot_positions"].values():
        assert positions.numel() >= 1
        assert (item["labels"][positions] == -100).all()
    # Everything before the first assistant token is ignored.
    first_label = torch.where(item["labels"] != -100)[0][0]
    assert (item["labels"][:first_label] == -100).all()


def test_truncation_drops_whole_turns():
    dialogue = Dialogue("d", "s:c", "c", "s", "train", [Scan("S1", "x", None, "RGB", "native_2d")], {}, [
        Turn("T1", "old words", "old answer", None, None, "S1"),
        Turn("T2", "new", "new answer", None, None, "S1"),
    ])
    loader = lambda _: (torch.zeros(3, 1, 4, 4), torch.ones(1, 4, 4, dtype=torch.bool), {})
    tokenizer = FakeTokenizer()
    full = DialogueDataset([dialogue], tokenizer, DatasetConfig(scan_loader=loader, max_len=100))[0]
    one_turn = Dialogue("d2", "s:c", "c", "s", "train", dialogue.scans, {}, [dialogue.turns[1]])
    short_len = DialogueDataset([one_turn], tokenizer, DatasetConfig(scan_loader=loader, max_len=100))[0]["input_ids"].numel()
    short = DialogueDataset([dialogue], tokenizer, DatasetConfig(scan_loader=loader, max_len=short_len))[0]
    assert len(short["turn_boundaries"]) == 1
    assert "T2" in [x["turn_id"] for x in short["actions"]] or not short["actions"]
