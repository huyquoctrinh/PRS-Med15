import json
from pathlib import Path

import pytest

from prs_med15.data.sources.mrmedseg import convert_mrmedseg, image_identity
from prs_med15.data.sources.our3d import convert_our3d
from prs_med15.data.validators import validate_dialogue


ROOT = Path("/data2/gdp2988/prs_med")


@pytest.mark.parametrize("folder", ["amos_3d", "totalseg_3d", "totalseg_mr_3d", "kits_3d"])
def test_one_generated_header_if_available(folder):
    paths = sorted((ROOT / "prsmed_g/dialogues" / folder).glob("dialogues_*.jsonl"))
    if not paths:
        pytest.skip(f"{folder} not available")
    with paths[0].open() as stream:
        line = stream.readline()
    if not line:
        pytest.skip(f"{folder} is still being written")
    dialogue = convert_our3d(json.loads(line))
    validate_dialogue(dialogue)


def test_one_mrmedseg_header_if_available():
    ijson = pytest.importorskip("ijson")
    root = ROOT / "external/MR-MedSeg/MR-MedSeg"
    path = root / "val.json"
    if not path.exists():
        pytest.skip("MR-MedSeg not available")
    with path.open("rb") as stream:
        record = next(ijson.items(stream, "item"))
    dialogue = convert_mrmedseg(record, root, "val")
    validate_dialogue(dialogue)


def test_mrmedseg_named_mr_sequence_filename():
    record = {"conversations": [{
        "from": "human",
        "value": "[IMAGE256:images/mr_t1--BraTS2019--HGG__case_1--x_0062.png] segment",
    }]}
    source, case, _, modality = image_identity(record)
    assert (source, case, modality) == ("mrmedseg:BraTS2019", "HGG__case_1", "MR")
