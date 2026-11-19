import pytest

from prs_med15.data.schema import Dialogue
from prs_med15.data.splits import assert_disjoint, drop_mrmedseg_leakage, normalise_kits_id, overlap_report


def dialogue(case, split="train", source="kits"):
    return Dialogue(case, f"{source}:{case}", case, source, split, [], {}, [])


def test_normalize_and_disjoint():
    assert normalise_kits_id("case12") == "case_00012"
    assert normalise_kits_id("case_00012") == "case_00012"
    assert_disjoint({"train": {"a"}, "val": {"b"}, "test": set(), "holdout": set()})
    with pytest.raises(ValueError):
        assert_disjoint({"train": {"a"}, "val": {"a"}})


def test_synthetic_leakage_drop_and_report():
    rows = [dialogue("case12"), dialogue("case13"), dialogue("case14", "val")]
    kept, report = drop_mrmedseg_leakage(rows, {"kits": {"val": {"case_00012"}, "test": set(), "holdout": set()}})
    assert [x.case_id for x in kept] == ["case13", "case14"]
    assert report == {"input": 3, "dropped": 1, "kept": 2, "forbidden_splits": ["val", "test"]}
    assert overlap_report(kept)["train_val"] == 0


def test_leakage_policy_benchmark_keeps_holdout_strict_drops_it():
    from prs_med15.data.splits import STRICT_FORBIDDEN_SPLITS
    rows = [dialogue("case12"), dialogue("case13")]
    splits = {"kits": {"val": set(), "test": {"case_00012"}, "holdout": {"case_00013"}}}
    kept, _ = drop_mrmedseg_leakage(rows, splits)
    assert [x.case_id for x in kept] == ["case13"]
    kept, _ = drop_mrmedseg_leakage(rows, splits, forbidden_splits=STRICT_FORBIDDEN_SPLITS)
    assert kept == []
