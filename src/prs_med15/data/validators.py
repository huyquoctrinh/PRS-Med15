"""Deterministic schema and protocol validation."""
from __future__ import annotations

from collections.abc import Iterable

from .schema import Dialogue
from .splits import check_patient_split_consistency


def validate_dialogue(dialogue: Dialogue) -> None:
    scans = {scan.scan_id for scan in dialogue.scans}
    known: set[str] = set()
    for turn in dialogue.turns:
        if turn.scan_id not in scans:
            raise ValueError(f"turn {turn.turn_id} references missing scan")
        if turn.assistant_text.count("[SEG]") != sum(a.op in {"create", "update"} for a in turn.actions):
            raise ValueError(f"turn {turn.turn_id}: [SEG]/action mismatch")
        created_this_turn = {a.entity_id for a in turn.actions if a.op == "create"}
        for ref in turn.reference_entity_ids:
            if ref not in known and ref not in created_this_turn:
                raise ValueError(f"turn {turn.turn_id}: REF to unknown entity {ref}")
        for action in turn.actions:
            if action.scan_id not in scans:
                raise ValueError("action references missing scan")
            for qualified in action.refs:
                entity = qualified.split(":", 1)[-1]
                if entity not in known:
                    raise ValueError(f"action REF to unknown entity {entity}")
            if action.op == "create":
                if action.entity_id in known:
                    raise ValueError("CREATE reuses entity")
                known.add(action.entity_id)
            if action.op in {"create", "update"}:
                obj = dialogue.objects.get(action.entity_id)
                if obj is None or (obj.label_id is None and not obj.mask_file and not obj.mask_path):
                    raise ValueError(f"action {action.entity_id} has no target mask reference")


def validate_manifest(dialogues: Iterable[Dialogue]) -> None:
    values = list(dialogues)
    check_patient_split_consistency(values)
    for dialogue in values:
        validate_dialogue(dialogue)
