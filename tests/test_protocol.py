import pytest

from prs_med15.protocol.parser import ProtocolError, parse_assistant
from prs_med15.protocol.serializer import serialize_mrmedseg, serialize_our3d


def test_our3d_create_ref_and_absent():
    record = {"conversations": [
        {"from": "human", "value": "q1"},
        {"from": "gpt", "value": "<think>A [SEG], B [SEG].</think> A [SEG].",
         "seg_objects": ["o1", "o2", "o1"], "intent": "ground"},
        {"from": "human", "value": "absent?"},
        {"from": "gpt", "value": "Not visible.", "intent": "absent", "seg_objects": []},
        {"from": "human", "value": "again"},
        {"from": "gpt", "value": "A [SEG].", "seg_objects": ["o1"], "intent": "ground"},
    ]}
    turns = serialize_our3d(record)
    assert [a["entity_id"] for a in turns[0].actions] == ["E1", "E2"]
    assert turns[0].text.count("[SEG]") == 2
    assert "<REF> S1:E1" in turns[0].text
    assert turns[1].actions == [] and "[SEG]" not in turns[1].text
    assert turns[2].text == "A <REF> S1:E1."


def test_mrmedseg_ind_and_ref_inside_create():
    record = {"conversations": [
        {"from": "human", "value": "[IMAGE256:images/ct_00--AMOS2022--amos_0001--x_1.png] first"},
        {"from": "gpt", "value": "[MASK-DECODE:masks/a.png]"},
        {"from": "human", "value": "[MASK-ENCODE:masks/a.png][BOX-ENCODE:masks/a.png] next", "ind": 0},
        {"from": "gpt", "value": "[REF-DECODE:masks/a.png][MASK-DECODE:masks/b.png]"},
    ]}
    pairs = serialize_mrmedseg(record)
    assert pairs[1][0].startswith("<REF> S1:E1")
    assert pairs[1][1].text == "<CREATE> <SCAN_REF> S1 <ENTITY> E2 <REF> S1:E1 [SEG] </CREATE>"
    assert pairs[1][1].reference_entity_ids == ["E1"]


def test_parser_validation_rejections():
    good = "<CREATE> <SCAN_REF> S1 <ENTITY> E1 [SEG] </CREATE>"
    assert parse_assistant(good, {"S1"}, [])[0].entity_id == "E1"
    cases = [
        ("<CREATE> <SCAN_REF> S2 <ENTITY> E1 [SEG] </CREATE>", "unknown_scan"),
        ("<CREATE> <SCAN_REF> S1 <ENTITY> E1 <REF> S1:E9 [SEG] </CREATE>", "unknown_entity_ref"),
        (good, "entity_reuse"),
        ("<CREATE> x", "unbalanced_tags"),
    ]
    for i, (text, code) in enumerate(cases):
        with pytest.raises(ProtocolError) as error:
            parse_assistant(text, {"S1"}, {"S1:E1"} if i == 2 else set())
        assert error.value.code == code


def test_our3d_same_turn_ref_is_not_a_reference():
    from prs_med15.protocol.serializer import serialize_our3d
    record = {"conversations": [
        {"from": "human", "value": "q1"},
        {"from": "gpt", "value": "<think>liver [SEG]</think>\nThe liver [SEG].", "seg_objects": ["o1", "o1"]},
        {"from": "human", "value": "q2"},
        {"from": "gpt", "value": "<think>liver [SEG] and spleen [SEG]</think>\nok", "seg_objects": ["o1", "o2"]},
    ]}
    t1, t2 = serialize_our3d(record)
    assert t1.reference_entity_ids == []
    assert t2.reference_entity_ids == ["E1"]
    assert [a["entity_id"] for a in t2.actions] == ["E2"]
