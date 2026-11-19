"""Deterministic conversion of source annotations into the bounded protocol."""
from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any

from .tokens import DETAIL_SLOT, GLOBAL_SLOT, REGION_SLOT


@dataclass
class SerializedTurn:
    text: str
    actions: list[dict[str, Any]] = field(default_factory=list)
    refs: list[str] = field(default_factory=list)
    reference_entity_ids: list[str] = field(default_factory=list)


def build_user_message(text: str, *, first_turn: bool, scan_id: str = "S1") -> str:
    prefix = f"<SCAN> {scan_id} <VIS>{GLOBAL_SLOT}</SCAN>\n" if first_turn else ""
    return f"{prefix}<DETAIL>{DETAIL_SLOT}\n{text}"


def build_memory_message(scan_id: str, entity_id: str) -> str:
    return f"<REGION> {scan_id}:{entity_id} <REGION_FEAT>{REGION_SLOT} </REGION>"


def serialize_our3d(record: dict[str, Any]) -> list[SerializedTurn]:
    """Serialize GPT turns while maintaining one dialogue-level entity registry."""
    object_entities: dict[str, str] = {}
    output: list[SerializedTurn] = []
    conversations = record.get("conversations", [])
    for conv in conversations:
        if conv.get("from") != "gpt":
            continue
        objects = iter(conv.get("seg_objects", []))
        actions: list[dict[str, Any]] = []
        refs: list[str] = []

        def replace(_: re.Match[str]) -> str:
            try:
                object_key = next(objects)
            except StopIteration as exc:
                raise ValueError("more [SEG] markers than seg_objects") from exc
            if object_key in object_entities:
                ref = f"S1:{object_entities[object_key]}"
                refs.append(ref)
                return f"<REF> {ref}"
            entity_id = f"E{len(object_entities) + 1}"
            object_entities[object_key] = entity_id
            marker = f"<CREATE> <SCAN_REF> S1 <ENTITY> {entity_id} [SEG] </CREATE>"
            actions.append({"op": "create", "scan_id": "S1", "entity_id": entity_id,
                            "object_key": object_key, "char_span": None, "refs": []})
            return marker

        text = re.sub(r"\[SEG\]", replace, conv.get("value", ""))
        try:
            next(objects)
        except StopIteration:
            pass
        else:
            raise ValueError("more seg_objects than [SEG] markers")
        cursor = 0
        for action in actions:
            needle = (f"<CREATE> <SCAN_REF> S1 <ENTITY> {action['entity_id']} "
                      "[SEG] </CREATE>")
            start = text.index(needle, cursor)
            action["char_span"] = [start, start + len(needle)]
            cursor = start + len(needle)
        created_here = {a["entity_id"] for a in actions}
        # reference_entity_ids lists only entities from EARLIER turns (same-turn REFs point back to
        # this turn's own CREATE and must not feed this turn's retrieval query).
        earlier = [r.split(":")[1] for r in refs if r.split(":")[1] not in created_here]
        output.append(SerializedTurn(text=text, actions=actions, refs=refs,
                                     reference_entity_ids=list(dict.fromkeys(earlier))))
    return output


_IMAGE = re.compile(r"\[IMAGE256:([^\]]+)\]")
_MASK_DECODE = re.compile(r"\[MASK-DECODE:([^\]]+)\]")
_REF_DECODE = re.compile(r"\[REF-DECODE:([^\]]+)\]")
_MASK_INPUT = re.compile(r"\[(?:MASK|BOX)-ENCODE:[^\]]+\]")


def serialize_mrmedseg(record: dict[str, Any]) -> list[tuple[str, SerializedTurn]]:
    """Return paired user/assistant turns and resolve MR-MedSeg round references."""
    path_entity: dict[str, str] = {}
    round_entity: dict[int, str] = {}
    result: list[tuple[str, SerializedTurn]] = []
    pending_user = ""
    pending_refs: list[str] = []
    round_index = 0
    for conv in record.get("conversations", []):
        if conv.get("from") == "human":
            pending_user = _MASK_INPUT.sub("", _IMAGE.sub("", conv.get("value", ""))).strip()
            pending_refs = []
            if conv.get("ind") is not None:
                entity = round_entity.get(int(conv["ind"]))
                if entity is None:
                    raise ValueError(f"MR-MedSeg ind references unknown round {conv['ind']}")
                pending_refs.append(entity)
                pending_user = f"<REF> S1:{entity} {pending_user}".strip()
            continue
        if conv.get("from") != "gpt":
            continue
        raw = conv.get("value", "")
        ref_paths = _REF_DECODE.findall(raw)
        local_ref_ids: list[str] = []
        for path in ref_paths:
            if path not in path_entity:
                raise ValueError(f"REF-DECODE references unknown mask: {path}")
            local_ref_ids.append(path_entity[path])
        raw = _REF_DECODE.sub("", raw)
        actions: list[dict[str, Any]] = []

        def decode(match: re.Match[str]) -> str:
            path = match.group(1)
            if path in path_entity:
                return f"<REF> S1:{path_entity[path]}"
            entity = f"E{len(path_entity) + 1}"
            path_entity[path] = entity
            refs = [f"S1:{e}" for e in local_ref_ids]
            ref_text = "".join(f" <REF> {ref}" for ref in refs)
            marker = f"<CREATE> <SCAN_REF> S1 <ENTITY> {entity}{ref_text} [SEG] </CREATE>"
            actions.append({"op": "create", "scan_id": "S1", "entity_id": entity,
                            "object_key": path, "char_span": None, "refs": refs})
            return marker

        text = _MASK_DECODE.sub(decode, raw)
        for action in actions:
            needle_start = f"<CREATE> <SCAN_REF> S1 <ENTITY> {action['entity_id']}"
            start = text.index(needle_start)
            end = text.index("</CREATE>", start) + len("</CREATE>")
            action["char_span"] = [start, end]
        if actions:
            round_entity[round_index] = actions[0]["entity_id"]
        refs = [f"S1:{e}" for e in local_ref_ids]
        result.append((pending_user, SerializedTurn(
            text=text, actions=actions, refs=refs,
            reference_entity_ids=list(dict.fromkeys(pending_refs + local_ref_ids)),
        )))
        round_index += 1
    return result
