import numpy as np
import pytest
from experiments.language_object_graph.objects import ObjectGraph
from experiments.role_aware_object_graph.parser import parse
from experiments.role_aware_object_graph.inference import forward,beams


def graph():
 objects={}
 for i,(x,name,typ) in enumerate([(-30,'White Church','Building'),(30,'Black Hall','Building'),(0,'','Building'),(5,'','Car'),(10,'','Car')]):
  objects[str(i)]={'id':i,'name':name,'object_type':typ,'position':[x,0,5],'dimension':[8,8,5],'contour':[[x-4,-4],[x+4,-4],[x+4,4],[x-4,4]]}
 return ObjectGraph(objects,[-50,-50,50,50])

def public():
 x,y=np.meshgrid(np.linspace(0,1,28),np.linspace(0,1,28));return dict(grid_xy=np.stack([x.ravel(),y.ravel()],1),static_logits=np.zeros(784),bounds=[-50,-50,50,50],pose_world=[0,0,50,0])

def test_name_outside_target_nominal_cannot_bind_target():
 l=parse('The gray car on Black Hall next to a red car.',graph())
 assert l['entities'][0]['object_type']=='Car' and not l['entities'][0]['named_ids']
 assert any(e['named_names']==('Black Hall',) and e['role']=='reference' for e in l['entities'])

def test_target_attributes_have_local_scope():
 l=parse('A gray building with a white car beside Black Hall.',graph())
 assert l['entities'][0]['colors']==('gray',)
 assert any(e['object_type']=='Car' and e['colors']==('white',) for e in l['entities'])

def test_nested_relations_have_reference_subject_and_alternative():
 l=parse('A gray car next to a red car behind Black Hall.',graph())
 f=next(f for f in l['factors'] if f['kind']=='behind');assert f['subject']!=0
 assert {h['subject'] for h in f['subject_hypotheses']}=={f['subject'],0}

def test_between_is_one_joint_factor_with_distinct_references():
 l=parse('The small house between White Church and Black Hall.',graph());f=l['factors'][0]
 assert f['kind']=='between' and len(f['anchors'])==2 and len(set(f['anchors']))==2

def test_leading_reference_this_subject():
 l=parse('Near Black Hall, this white car is next to a red car.',graph());assert l['entities'][0]['object_type']=='Car' and l['entities'][0]['colors']==('white',)

def test_direct_named_target_remains_named():
 l=parse('Go to White Church near a gray building.',graph());assert l['entities'][0]['named_names']==('White Church',)

def test_group_target_unknown_exact_geometry_fallback():
 p=public();r=forward(graph(),'Multiple buildings near Black Hall',p,method='A4',weight=.5)
 assert r['language']['confidence']==0 and np.array_equal(r['logits'],p['static_logits'])

def test_relative_frame_and_unresolved_between_reported():
 l=parse('The house left of Black Hall between two houses',graph());assert len(l['factors'][0]['frame_hypotheses'])==3
 assert any(f['kind']=='between' and len(f['anchors'])==2 for f in l['factors'])

def test_pruned_beam_mass_retained_as_unknown():
 l={'hypotheses':[{'target':0,'probability':1}], 'factors':[{'subject_hypotheses':[{'subject':0,'probability':.5},{'subject':1,'probability':.5}]} for _ in range(8)]}
 b=beams(l,limit=8);assert abs(sum(h['probability'] for h in b)-1)<1e-9;assert b[-1]['null'] and b[-1]['probability']>.9

def test_dependency_same_text_required():
 with pytest.raises(ValueError):parse('a house',graph(),backend='dependency',dependency={'instruction':'another text'})

def test_gt_isolation_and_all_grid_cells_retained():
 p=public()
 with pytest.raises(ValueError):forward(graph(),'a house near Black Hall',{**p,'target_id':'0'},method='A4',weight=.5)
 r=forward(graph(),'a house near Black Hall',p,method='A4',weight=.5)
 assert len(r['logits'])==784 and np.isfinite(r['logits']).all() and np.isclose(r['heatmap'].sum(),1)


def test_unknown_language_keeps_geometry_exact():
 p=public();r=forward(graph(),'It is that one.',p,method='A4',weight=.5);assert np.array_equal(r['logits'],p['static_logits'])
