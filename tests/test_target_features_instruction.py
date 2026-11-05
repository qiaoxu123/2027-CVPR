import pytest
from graphnav.target_features_instruction import content_instruction

@pytest.mark.parametrize('text,expected',[
 ('Navigate to the white building north of The Food Warehouse.', 'the white building north of The Food Warehouse.'),
 ('Fly to the third blue car in the parking lot behind One Stop, not the road. Stop beside it.', 'the third blue car in the parking lot behind One Stop, not the road. Stop beside it.'),
 ('The building between One Stop and Church.', 'The building between One Stop and Church.'),
 ('Navigate to One Stop.', 'One Stop.'),
 ('Go north of One Stop, then turn right.', 'Go north of One Stop, then turn right.'),
 ('Go to', 'Go to'),
])
def test_preserve_complete_content(text,expected):
    assert content_instruction(text)==expected


def test_reject_empty():
    with pytest.raises(ValueError): content_instruction('   ')


def test_preparation_changes_no_labels_or_public_polygons(tmp_path):
    import json
    from scripts.prepare_target_features_text import prepare
    row=dict(map_name='birmingham_block_1',instruction='Navigate to the third white building north of One Stop.',target_xy=[.2,.3],world_bounds=[0,0,100,100],landmarks=[dict(id='public',name='One Stop',contour=[[0,0],[.2,0],[.2,.2]])],dataset_version='revised',match_mode='fuzzy')
    source=tmp_path/'bound.jsonl';original=tmp_path/'native.jsonl'
    source.write_text(json.dumps(row)+'\n');original.write_text(json.dumps({**row,'landmarks':[]})+'\n')
    out=tmp_path/'out';prepare(source,original,out,expected=1)
    result=json.loads((out/'target_features_landmark_relation.jsonl').read_text())
    assert result['instruction']=='the third white building north of One Stop.'
    assert {k:v for k,v in row.items() if k!='instruction'}=={k:v for k,v in result.items() if k!='instruction'}
    row['target_xy']=[.9,.9];source.write_text(json.dumps(row)+'\n')
    with pytest.raises(ValueError,match='Misaligned'):
        prepare(source,original,tmp_path/'bad',expected=1)
