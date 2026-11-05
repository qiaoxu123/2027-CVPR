import json
import pytest
from scripts.evaluate_closedloop import resume_jsonl_rows


def test_partial_final_write_preserved_but_not_counted(tmp_path):
    p=tmp_path/'predictions.jsonl';valid=json.dumps({'index':0,'metrics':{'SR':0}})+'\n';p.write_text(valid+'{"index":1,')
    rows=resume_jsonl_rows(p)
    assert len(rows)==1 and rows[0]['index']==0
    assert p.read_text()==valid
    assert (tmp_path/'predictions.jsonl.incomplete_tail').read_text()=='{"index":1,'


def test_corruption_in_complete_line_is_not_silently_dropped(tmp_path):
    p=tmp_path/'predictions.jsonl';p.write_text('{bad}\n')
    with pytest.raises(json.JSONDecodeError):resume_jsonl_rows(p)
