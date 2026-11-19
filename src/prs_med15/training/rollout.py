"""History selection and accounting for turn-ordered training rollouts."""
from __future__ import annotations

from dataclasses import dataclass

import torch


def choose_dialogue_history(
    mode: str,
    oracle_fraction: float = 0.5,
    generator: torch.Generator | None = None,
) -> str:
    """Choose once per dialogue, as required by mixed-history training."""
    if mode == "oracle":
        return "oracle"
    if mode == "predicted":
        return "predicted"
    if mode != "mixed":
        raise ValueError("history mode must be oracle, predicted, or mixed")
    sample = torch.rand((), generator=generator)
    return "oracle" if bool(sample < oracle_fraction) else "predicted"


def choose_history_mask(
    oracle: torch.Tensor | None,
    predicted: torch.Tensor,
    selection: str,
) -> tuple[torch.Tensor, str]:
    if selection == "oracle" and oracle is not None:
        return oracle.detach(), "oracle"
    if selection not in {"oracle", "predicted"}:
        raise ValueError("selection must be oracle or predicted")
    return predicted.detach(), "predicted"


@dataclass
class HistoryUsage:
    oracle: int = 0
    predicted: int = 0
    corrupted: int = 0

    def add(
        self,
        provenance: str,
        mask: torch.Tensor,
        oracle_mask: torch.Tensor | None,
    ) -> None:
        if provenance == "oracle":
            self.oracle += 1
            return
        self.predicted += 1
        if oracle_mask is not None:
            expected = oracle_mask.to(mask.device).bool()
            if not torch.equal(mask.bool(), expected):
                self.corrupted += 1

    def fractions(self) -> dict[str, float]:
        total = self.oracle + self.predicted
        if total == 0:
            return {"oracle": 0.0, "predicted": 0.0, "corrupted": 0.0}
        return {
            "oracle": self.oracle / total,
            "predicted": self.predicted / total,
            "corrupted": self.corrupted / total,
        }


def teacher_forced(entries):
    return list(entries)


def predicted_history(entries):
    return [entry.detach() if hasattr(entry, "detach") else entry for entry in entries]
