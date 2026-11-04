import json
from pathlib import Path
import numpy as np
import pytest
from scripts.prepare_strict_direct_direction import decide
from scripts.evaluate_strict_direct_direction import assert_public_inputs
from scripts.score_strict_direct_direction import paired
from scripts.score_direction_belief import score

def test_text_scope_not_bag_of_words():
 assert not decide({'index':4})[0]  # car inside ordinal row west of a landmark
 assert decide({'index':20,'word':'north'},['Building'])[0]  # explicit target north; row is extra description
 assert not decide({'index':16})[0]  # facing, not location
 assert not decide({'index':93})[0]  # across from building north of landmark

def test_mixed_anchor_abstains_without_gt():
 row={'index':80,'word':'north'}
 assert not decide(row,['Building','TrafficRoad'])[0]
 row['target_world']=[9999,-9999]
 assert not decide(row,['Building','TrafficRoad'])[0]

@pytest.mark.parametrize('key',['target_world','target_ids','target_type'])
def test_targets_forbidden_in_model_inputs(key):
 with pytest.raises(ValueError):assert_public_inputs([{'word':'north',key:None}])
 assert_public_inputs([{'word':'north','public_object_ids':['5']}])

def test_scoring_uses_original_point_not_anchor():
 anchor=np.zeros((1,28,28));anchor[0,14,14]=1
 target=anchor.copy();target[0]=0;target[0,16,14]=1
 span=np.array([[280.,280.]]);gt=np.array([[145.,165.]]);ids=np.array([0])
 v=score(target,span,gt,anchor,ids)
 assert v['R20']==1 and v['NE']==0 and v['GT20_mass']==1
 assert v['direction_sector45']==1
 assert paired({'per_sample_NE':[30,10]},{'per_sample_NE':[10,30]})=={'corrected':1,'damaged':1,'net':0}

def test_sealed_selection_and_prediction_are_unchanged():
 from graphnav.checkpointing import file_sha256
 root=Path(__file__).resolve().parents[1]/'reports/strict_direct_direction'
 if not root.exists():pytest.skip('Audit artifacts not installed')
 s=json.loads((root/'prediction_seal.json').read_text())
 assert file_sha256(root/'public.json')==s['cohort_sha256']
 assert file_sha256(root/'selection.json')==s['selection_sha256']
 assert file_sha256(root/'probabilities.npz')==s['prediction_sha256']
 assert s['target_labels_read'] is False
 assert file_sha256(root/'prediction_producer.py')==s['code_sha256']
 rows=json.loads((root/'public.json').read_text());assert_public_inputs(rows)
 assert len(rows)==97
 assert all(x['word'] in ('north','south') for x in rows)
