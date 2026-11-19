"""Strict protocol parser with controlled, machine-readable failures."""
from __future__ import annotations

from dataclasses import dataclass
import re


class ProtocolError(ValueError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class ParsedAction:
    op: str
    scan_id: str
    entity_id: str
    refs: tuple[str, ...]
    char_span: tuple[int, int]


_TAGS = re.compile(r"</?(?:CREATE|UPDATE|SHOW)>")
_BLOCK = re.compile(r"<(CREATE|UPDATE|SHOW)>\s*(.*?)\s*</\1>", re.DOTALL)
_REF = re.compile(r"<REF>\s+(S[1-9]\d*:E[1-9]\d*)")


def parse_assistant(text: str, allowed_scans: set[str], existing_entities: set[str]) -> list[ParsedAction]:
    covered = [False] * len(text)
    actions: list[ParsedAction] = []
    created = set(existing_entities)
    for match in _BLOCK.finditer(text):
        for i in range(match.start(), match.end()):
            covered[i] = True
        op, body = match.group(1).lower(), match.group(2)
        scan_match = re.search(r"<SCAN_REF>\s+(S[1-9]\d*)", body)
        entity_tag = "ENTITY" if op == "create" else "ENTITY_REF"
        entity_match = re.search(rf"<{entity_tag}>\s+(E[1-9]\d*)", body)
        if not scan_match or not entity_match:
            raise ProtocolError("malformed_action", f"{op} lacks scan/entity field")
        scan_id, entity_id = scan_match.group(1), entity_match.group(1)
        if scan_id not in allowed_scans:
            raise ProtocolError("unknown_scan", scan_id)
        qualified = f"{scan_id}:{entity_id}"
        refs = tuple(r.group(1) for r in _REF.finditer(body))
        for ref in refs:
            if ref not in created:
                raise ProtocolError("unknown_entity_ref", ref)
        if op == "create":
            if qualified in created:
                raise ProtocolError("entity_reuse", qualified)
            if body.count("[SEG]") != 1:
                raise ProtocolError("seg_count", "CREATE requires exactly one [SEG]")
            created.add(qualified)
        elif op == "update":
            if qualified not in created:
                raise ProtocolError("unknown_entity", qualified)
            if body.count("[SEG]") != 1:
                raise ProtocolError("seg_count", "UPDATE requires exactly one [SEG]")
        elif qualified not in created:
            raise ProtocolError("unknown_entity", qualified)
        actions.append(ParsedAction(op, scan_id, entity_id, refs, (match.start(), match.end())))
    for tag in _TAGS.finditer(text):
        if not all(covered[tag.start():tag.end()]):
            raise ProtocolError("unbalanced_tags", tag.group(0))
    valid_refs = list(_REF.finditer(text))
    if text.count("<REF>") != len(valid_refs):
        raise ProtocolError("malformed_ref", "REF requires a qualified S#:E# alias")
    # A reference outside an action may only use entities that existed before its text position.
    initial = set(existing_entities)
    for ref in valid_refs:
        if any(covered[ref.start():ref.end()]):
            continue
        available = initial | {
            f"{action.scan_id}:{action.entity_id}" for action in actions
            if action.op == "create" and action.char_span[1] <= ref.start()
        }
        if ref.group(1) not in available:
            raise ProtocolError("unknown_entity_ref", ref.group(1))
    for ref in _REF.finditer(text):
        if ref.group(1) not in created:
            raise ProtocolError("unknown_entity_ref", ref.group(1))
    return actions
