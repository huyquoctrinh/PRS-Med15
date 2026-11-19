"""Adapter for the four generated 3D dialogue sources."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

from ...protocol.serializer import serialize_our3d
from ..schema import Action, Dialogue, ObjectRecord, Scan, Turn

SOURCE_MAP = {"amos22": "amos", "totalseg": "totalseg", "totalseg_mr": "totalseg_mr", "kits23": "kits"}


def convert_our3d(record: dict[str, Any]) -> Dialogue:
    dataset = record["dataset"]
    if dataset not in SOURCE_MAP:
        raise ValueError(f"unsupported generated dataset: {dataset}")
    source = SOURCE_MAP[dataset]
    serialized = serialize_our3d(record)
    human = [x.get("value", "") for x in record.get("conversations", []) if x.get("from") == "human"]
    gpt = [x for x in record.get("conversations", []) if x.get("from") == "gpt"]
    if not (len(human) == len(gpt) == len(serialized)):
        raise ValueError("generated dialogue has unpaired turns")
    object_entities: dict[str, str] = {}
    turns: list[Turn] = []
    for i, (user, raw, ser) in enumerate(zip(human, gpt, serialized, strict=True)):
        actions = [Action(**a) for a in ser.actions]
        object_entities.update({a.object_key: a.entity_id for a in actions if a.object_key})
        turns.append(Turn(
            turn_id=f"T{i + 1}", user_text=user, assistant_text=ser.text,
            intent=raw.get("intent"), operation="create" if actions else None, scan_id="S1",
            actions=actions, reference_entity_ids=ser.reference_entity_ids,
            answerability=raw.get("answerability", "answerable"),
        ))
    objects: dict[str, ObjectRecord] = {}
    for object_key, entity_id in object_entities.items():
        raw = record.get("objects", {}).get(object_key, {})
        mask_file = raw.get("mask_file")
        objects[entity_id] = ObjectRecord(
            object_key=object_key, canonical=raw.get("canonical", object_key),
            label_id=raw.get("label_id"), mask_file=mask_file,
            mask_path=(f"{record['label']}/{mask_file}" if mask_file else None),
        )
    return Dialogue(
        dialogue_id=record["id"], patient_group_id=f"{source}:{record['case']}",
        case_id=record["case"], source=source, split=record["split"],
        scans=[Scan("S1", record["image"], record.get("label"), record["modality"], "volume", None)],
        objects=objects, turns=turns,
    )


def iter_our3d(path: str | Path) -> Iterator[Dialogue]:
    """Stream a JSONL file without retaining the source corpus."""
    with Path(path).open() as stream:
        for line in stream:
            if line.strip():
                yield convert_our3d(json.loads(line))
