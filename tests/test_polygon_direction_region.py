import numpy as np
import pytest
from graphnav.polygon_direction_region import polygon_direction_region,validate_public_record
from scripts.scan_polygon_direction import retrieve_candidates

def square(x0,y0,x1,y1):return [[x0,y0],[x1,y0],[x1,y1],[x0,y1]]

@pytest.mark.parametrize('word,u',[('north',(0,1)),('south',(0,-1)),('east',(1,0)),('west',(-1,0))])
def test_polygon_ray_respects_axes_and_continuous_distances(word,u):
 out=polygon_direction_region([square(40,40,60,60)],[0,0,100,100],word,mode='ray')
 assert out['available'] and out['probabilities'].sum()==pytest.approx(1.)
 xy=out['world_grid'];p=out['probabilities'];d=(xy-[50,50])@np.array(u)
 assert p[d<0].sum()==0
 assert np.count_nonzero(p)>10
 assert p[(d>30)].sum()>0

def test_full_fragment_region_is_not_centroid_shift():
 p=[square(5,20,20,30),square(75,20,90,30)]
 out=polygon_direction_region(p,[0,0,100,100],'north',mode='ray');pr=out['probabilities'];xy=out['world_grid']
 assert pr[(xy[...,0]<25)&(xy[...,1]>35)].sum()>0
 assert pr[(xy[...,0]>70)&(xy[...,1]>35)].sum()>0
 assert pr[(xy[...,0]>35)&(xy[...,0]<60)].sum()==0

def test_cone_adds_lateral_support_but_not_reverse_side():
 p=[square(40,40,60,60)];ray=polygon_direction_region(p,[0,0,100,100],'north',mode='ray');cone=polygon_direction_region(p,[0,0,100,100],'north',mode='cone');xy=ray['world_grid']
 assert np.count_nonzero(cone['probabilities'])>np.count_nonzero(ray['probabilities'])
 assert cone['probabilities'][xy[...,1]<40].sum()==0
 assert cone['probabilities'][(xy[...,0]>40)&(xy[...,0]<60)&(xy[...,1]>40)&(xy[...,1]<60)].sum()==0

def test_boundary_without_legal_region_abstains():
 out=polygon_direction_region([square(0,90,100,100)],[0,0,100,100],'north')
 assert not out['available'] and out['probabilities'].sum()==0

@pytest.mark.parametrize('k',['target_world','target_ids','target_type'])
def test_target_metadata_rejected(k):
 with pytest.raises(ValueError):validate_public_record({k:[1,2]})

def test_scan_rejects_proper_cardinal_name_and_keeps_real_predicate():
 cat={'map':{'1':{'name':'West House','object_type':'Building'}}}
 base=dict(episode=0,map_name='map',landmarks=[],relations=[],direct_relations=[],target={'category':'Car'},ambiguity=[])
 rows=[dict(base,original_text='A car near West House.')]
 candidates,scanned=retrieve_candidates(rows,cat)
 assert not candidates and len(scanned)==1

def test_fragment_order_and_coordinate_translation_do_not_change_prediction():
 polygons=[square(5,20,20,30),square(75,20,90,30)]
 a=polygon_direction_region(polygons,[0,0,100,100],'north')
 b=polygon_direction_region(polygons[::-1],[0,0,100,100],'north')
 delta=np.array([123.,-90.]);c=polygon_direction_region([(np.array(p)+delta).tolist() for p in polygons],[123,-90,223,10],'north')
 assert np.allclose(a['probabilities'],b['probabilities'])
 assert np.allclose(a['probabilities'],c['probabilities'])

def test_public_polygon_rotation_matches_declared_word_axes():
 p=[square(40,40,60,60)];north=polygon_direction_region(p,[0,0,100,100],'north',mode='ray')['probabilities'];east=polygon_direction_region(p,[0,0,100,100],'east',mode='ray')['probabilities']
 assert np.allclose(np.rot90(north),east)

def test_invalid_public_polygon_is_not_repaired_using_target():
 with pytest.raises(ValueError):polygon_direction_region([[[0,0],[10,10],[0,10],[10,0]]],[0,0,100,100],'north')

def test_selection_drops_intermediate_region_and_accepts_parking_target_itself():
 from scripts.select_polygon_direction import decide_candidate
 base=dict(episode=1,word='north',landmark_name='Example',public_types=['Building'],name_binding='existing_text_only_catalog_span')
 assert not decide_candidate(dict(base,candidate_id=15),{})[0]
 assert decide_candidate(dict(base,candidate_id=285),{})[0]  # parking area is target, north of Keynes House
 assert not decide_candidate(dict(base,candidate_id=404,word='west'),{})[0]

def test_sealed_groups_and_models_have_no_confirmation_exposure():
 import json,hashlib
 from pathlib import Path
 from graphnav.checkpointing import file_sha256
 o=Path(__file__).resolve().parents[1]/'reports/polygon_direction_region'
 if not (o/'prediction_seal.json').exists():pytest.skip('Experiment artifacts unavailable')
 seal=json.loads((o/'prediction_seal.json').read_text());assert file_sha256(o/'public.json')==seal['public_sha256'];assert file_sha256(o/'predictions.npz')==seal['prediction_sha256']
 for p,expected in seal['code_sha256'].items():assert file_sha256(o.parents[1]/p)==expected
 groups=json.loads((o/'groups.json').read_text());seen={r['group'] for r in groups if r['previous_exposure']};assert not any(r['group'] in seen for r in groups if r['subset'] in ('dev','confirm'))
 rows=json.loads((o/'public.json').read_text());episode_sets={s:{r['episode'] for r in rows if r['subset']==s} for s in ['fit','dev','confirm']};assert not episode_sets['fit']&episode_sets['confirm'] and not episode_sets['dev']&episode_sets['confirm']
 for r in rows:validate_public_record(r)
