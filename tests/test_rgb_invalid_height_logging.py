import io
import json
from scripts.evaluate_observed_rgb_matching import record_invalid_height


def test_invalid_height_preserves_denominator_in_jsonl():
    rows=[]
    stream=io.StringIO()
    record_invalid_height(rows,stream,index=7,step=3,map_name="example",height=-0.5)
    persisted=[json.loads(line) for line in stream.getvalue().splitlines()]
    assert persisted == rows
    assert len(persisted) == 1
    assert persisted[0]["eligible"] is False
    assert persisted[0]["reason"] == "nonpositive_height"
    assert persisted[0]["height_m"] == -0.5
