"""Bounded structural vocabulary; scan/entity aliases remain ordinary text."""
from __future__ import annotations

import re

STRUCT_TOKENS = [
    "<SCAN>", "</SCAN>", "<VIS>", "<DETAIL>", "<REGION>", "</REGION>",
    "<REGION_FEAT>", "<CREATE>", "</CREATE>", "<UPDATE>", "</UPDATE>",
    "<SHOW>", "</SHOW>", "<SCAN_REF>", "<ENTITY>", "<ENTITY_REF>", "<REF>", "[SEG]",
]

GLOBAL_SLOT = "{GLOBAL}"
DETAIL_SLOT = "{DETAIL}"
REGION_SLOT = "{REGION}"
SLOT_TOKENS = (GLOBAL_SLOT, DETAIL_SLOT, REGION_SLOT)


def scan_alias(index: int) -> str:
    if index < 1:
        raise ValueError("scan aliases are one-based")
    return f"S{index}"


def entity_alias(index: int) -> str:
    if index < 1:
        raise ValueError("entity aliases are one-based")
    return f"E{index}"


def alias_index(alias: str, prefix: str) -> int:
    match = re.fullmatch(rf"{re.escape(prefix)}([1-9][0-9]*)", alias)
    if not match:
        raise ValueError(f"invalid {prefix} alias: {alias}")
    return int(match.group(1))


GRAMMAR = """
assistant := text (action | reference | text)*
action    := CREATE | UPDATE | SHOW
CREATE    := <CREATE> <SCAN_REF> S<n> <ENTITY> E<n> (REF)* [SEG] </CREATE>
UPDATE    := <UPDATE> <SCAN_REF> S<n> <ENTITY_REF> E<n> (REF)* [SEG] </UPDATE>
SHOW      := <SHOW> <SCAN_REF> S<n> <ENTITY_REF> E<n> </SHOW>
REF       := <REF> S<n>:E<n>
""".strip()
