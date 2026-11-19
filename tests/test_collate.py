import torch
import pytest

from prs_med15.data.collate import collate_dialogues


def item(kind, depth):
    return {"input_kind": kind, "scans": [torch.ones(3, depth, 4, 5)],
            "valid_masks": [torch.ones(depth, 4, 5, dtype=torch.bool)],
            "input_ids": torch.tensor([2, 3]), "labels": torch.tensor([-100, 3]), "actions": []}


def test_collate_depth_and_token_padding():
    batch = collate_dialogues([item("volume", 1), item("volume", 3)], 0)
    assert batch["pixel_values"].shape == (2, 3, 3, 4, 5)
    assert not batch["pixel_valid_mask"][0, 1:].any()
    with pytest.raises(ValueError):
        collate_dialogues([item("volume", 1), item("native_2d", 1)])
