import json
from pathlib import Path
import pytest
from scripts.verify_native_navigation_result import verify


ROOT=Path('aerialvg_native_temporal_navigation_seen128_20261101')
FROZEN=Path('aerialvg_native_temporal_frozen_seen_candidate_20261101.json')


def test_complete_subset_cannot_pass_full_split_verification():
    if not (ROOT/'metrics.json').exists():pytest.skip('Completed actual native pilot required')
    with pytest.raises(ValueError,match='Full evaluation required'):
        verify(ROOT,FROZEN,128,require_full=True)


def test_full_declared_count_rejects_partial_trace_before_reporting_success(tmp_path):
    if not (ROOT/'metrics.json').exists():pytest.skip('Completed actual native pilot required')
    for name in ['metrics.json','native_experiment.json','episodes.json']:
        (tmp_path/name).write_bytes((ROOT/name).read_bytes())
    (tmp_path/'predictions.jsonl').write_text((ROOT/'predictions.jsonl').read_text().splitlines()[0]+'\n')
    with pytest.raises(ValueError,match='Incomplete rollout set'):
        verify(tmp_path,FROZEN,128)
    assert not (tmp_path/'verification.json').exists()
