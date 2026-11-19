"""Stable JSON-serializable Phase 1 dialogue schema."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from typing import Any, Literal


@dataclass
class Scan:
    scan_id: str
    image_path: str
    label_ref: str | None
    modality: Literal["CT", "MR", "RGB"]
    input_kind: Literal["volume", "slab", "native_2d"]
    geometry_ref: str | dict[str, Any] | None = None


@dataclass
class Action:
    op: Literal["create", "update", "show"]
    scan_id: str
    entity_id: str
    object_key: str | None = None
    char_span: list[int] | None = None
    refs: list[str] = field(default_factory=list)


@dataclass
class ObjectRecord:
    object_key: str
    canonical: str
    label_id: int | None = None
    mask_file: str | None = None
    mask_path: str | None = None


@dataclass
class Turn:
    turn_id: str
    user_text: str
    assistant_text: str
    intent: str | None
    operation: str | None
    scan_id: str
    actions: list[Action] = field(default_factory=list)
    reference_entity_ids: list[str] = field(default_factory=list)
    reference_turn_ids: list[str] = field(default_factory=list)
    answerability: str = "answerable"


@dataclass
class Dialogue:
    dialogue_id: str
    patient_group_id: str
    case_id: str
    source: str
    split: str
    scans: list[Scan]
    objects: dict[str, ObjectRecord]
    turns: list[Turn]

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)

    @classmethod
    def from_json(cls, text: str | dict[str, Any]) -> "Dialogue":
        value = json.loads(text) if isinstance(text, str) else text
        return cls(
            dialogue_id=value["dialogue_id"], patient_group_id=value["patient_group_id"],
            case_id=value["case_id"], source=value["source"], split=value["split"],
            scans=[Scan(**x) for x in value["scans"]],
            objects={k: ObjectRecord(**v) for k, v in value.get("objects", {}).items()},
            turns=[Turn(**{**t, "actions": [Action(**a) for a in t.get("actions", [])]})
                   for t in value.get("turns", [])],
        )
