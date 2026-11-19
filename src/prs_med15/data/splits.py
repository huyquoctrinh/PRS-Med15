"""Patient-level split and leakage utilities."""
from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any, Iterable

from .schema import Dialogue


def normalise_kits_id(case: str) -> str:
    match = re.fullmatch(r"case_?(\d+)", case, re.IGNORECASE)
    return f"case_{int(match.group(1)):05d}" if match else case


def load_split_file(path: str | Path, source: str | None = None) -> dict[str, set[str]]:
    raw = json.loads(Path(path).read_text())
    result: dict[str, set[str]] = {}
    for split in ("train", "val", "test", "holdout"):
        values = raw.get(split, [])
        if source == "kits" or "kits" in Path(path).stem:
            values = [normalise_kits_id(x) for x in values]
        result[split] = set(values)
    return result


def assert_disjoint(splits: dict[str, set[str]]) -> None:
    names = list(splits)
    for i, left in enumerate(names):
        for right in names[i + 1:]:
            overlap = splits[left] & splits[right]
            if overlap:
                raise ValueError(f"split overlap {left}/{right}: {sorted(overlap)[:5]}")


# Default = MR-MedSeg benchmark protocol (user decision 2026-11-16): MR-MedSeg train is kept as
# published except conversations whose patients are in OUR 3D val/test. MR-MedSeg itself splits by
# slice, so its test patients ("holdout" for us) also occur in its train set; pass
# forbidden_splits=("val", "test", "holdout") for the strict patient-level variant.
BENCHMARK_FORBIDDEN_SPLITS = ("val", "test")
STRICT_FORBIDDEN_SPLITS = ("val", "test", "holdout")


def drop_mrmedseg_leakage(dialogues: Iterable[Dialogue], our_splits: dict[str, dict[str, set[str]]],
                          forbidden_splits: tuple[str, ...] = BENCHMARK_FORBIDDEN_SPLITS,
                          ) -> tuple[list[Dialogue], dict[str, Any]]:
    kept: list[Dialogue] = []
    counts = {
        "input": 0,
        "dropped": 0,
        "kept": 0,
        "forbidden_splits": list(forbidden_splits),
    }
    for dialogue in dialogues:
        counts["input"] += 1
        source = dialogue.source.split(":", 1)[0]
        case = normalise_kits_id(dialogue.case_id) if source == "kits" else dialogue.case_id
        forbidden = set().union(*(our_splits.get(source, {}).get(k, set()) for k in forbidden_splits))
        if dialogue.split == "train" and case in forbidden:
            counts["dropped"] += 1
        else:
            kept.append(dialogue)
            counts["kept"] += 1
    return kept, counts


def overlap_report(dialogues: Iterable[Dialogue]) -> dict[str, int]:
    by_split: dict[str, set[str]] = {}
    for dialogue in dialogues:
        by_split.setdefault(dialogue.split, set()).add(dialogue.patient_group_id)
    report: dict[str, int] = {}
    names = sorted(by_split)
    for i, left in enumerate(names):
        for right in names[i + 1:]:
            report[f"{left}_{right}"] = len(by_split[left] & by_split[right])
    return report


def check_patient_split_consistency(dialogues: Iterable[Dialogue]) -> None:
    seen: dict[str, str] = {}
    for dialogue in dialogues:
        old = seen.setdefault(dialogue.patient_group_id, dialogue.split)
        if old != dialogue.split:
            raise ValueError(f"patient {dialogue.patient_group_id} occurs in {old} and {dialogue.split}")
