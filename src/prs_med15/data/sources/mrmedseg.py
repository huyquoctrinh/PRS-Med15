"""Streaming-friendly adapter for one MR-MedSeg conversation."""
from __future__ import annotations

from pathlib import Path
import re
from typing import Any, Iterator

from ...protocol.serializer import serialize_mrmedseg
from ..schema import Action, Dialogue, ObjectRecord, Scan, Turn
from ..splits import normalise_kits_id

_IMAGE = re.compile(r"\[IMAGE256:([^\]]+)\]")
_SOURCE = {"Totalsegmentator_dataset": "totalseg", "AMOS2022": "amos",
           "KiTS": "kits", "KiTS2021": "kits"}


def image_identity(record: dict[str, Any]) -> tuple[str, str, str, str]:
    first = next((x.get("value", "") for x in record.get("conversations", []) if x.get("from") == "human"), "")
    match = _IMAGE.search(first)
    if not match:
        raise ValueError("MR-MedSeg record has no IMAGE256 tag")
    image_rel = match.group(1)
    parts = Path(image_rel).stem.split("--")
    if len(parts) < 4:
        raise ValueError(f"unrecognised MR-MedSeg image name: {image_rel}")
    modality_raw, raw_source, case = parts[:3]
    source = _SOURCE.get(raw_source, f"mrmedseg:{raw_source}")
    if source == "kits":
        case = normalise_kits_id(case)
    modality_name = modality_raw.lower()
    if modality_name.startswith("ct"):
        modality = "CT"
    elif modality_name.startswith(("mr", "mri")):
        modality = "MR"
    else:
        modality = "RGB"
    return source, case, image_rel, modality


def convert_mrmedseg(record: dict[str, Any], root: str | Path, split: str) -> Dialogue:
    root = Path(root)
    source, case, image_rel, modality = image_identity(record)
    pairs = serialize_mrmedseg(record)
    objects: dict[str, ObjectRecord] = {}
    turns: list[Turn] = []
    for i, (user, ser) in enumerate(pairs):
        actions = [Action(**a) for a in ser.actions]
        for action in actions:
            if action.object_key:
                objects[action.entity_id] = ObjectRecord(
                    object_key=action.object_key, canonical=Path(action.object_key).stem,
                    mask_path=str(root / action.object_key),
                )
        turns.append(Turn(
            turn_id=f"T{i + 1}", user_text=user, assistant_text=ser.text, intent="segmentation",
            operation="create" if actions else None, scan_id="S1", actions=actions,
            reference_entity_ids=ser.reference_entity_ids,
        ))
    patient_source = source if not source.startswith("mrmedseg:") else source
    dialogue_id = f"mrmedseg_{source.replace(':', '_')}_{case}_{Path(image_rel).stem}"
    return Dialogue(
        dialogue_id=dialogue_id, patient_group_id=f"{patient_source}:{case}", case_id=case,
        source=source, split=split,
        scans=[Scan("S1", str(root / image_rel), None, modality, "native_2d",
                    {"unknown_geometry": True})], objects=objects, turns=turns,
    )


def iter_mrmedseg(path: str | Path, root: str | Path, split: str) -> Iterator[Dialogue]:
    """Stream the 283 MB training JSON exactly once via ijson."""
    try:
        import ijson
    except ImportError as exc:  # pragma: no cover - environment dependency is explicit
        raise RuntimeError("ijson is required for streaming MR-MedSeg") from exc
    with Path(path).open("rb") as stream:
        for record in ijson.items(stream, "item"):
            yield convert_mrmedseg(record, root, split)
