import copy,pytest
from shapely.geometry import Polygon
from common import *
from infer import generate,routing_ids

def obj(i,name,cat,xy,contour):return {'id':i,'name':name,'category':cat,'position':xy,'contour':contour}
def fixture():
 return {'occurrence_id':'x','cluster':'m:target','map':'m','instruction':'This is a building. If you stand at Boat Club facing the stream, it is on your right.','word':'right','direction_span':[77,82],'reference_ids':['r'],'reference_text':'Boat Club','reference_span':[35,44],'subject_ids':['target'],'parse_tier':'high','parse_confidence':.97,'start_xy':[30,30],'mentioned':[{'start':35,'end':44,'name':'boat club','resolved_ids':['r']}],'scene':{'r':obj('r','Boat Club','Building',[0,0],[[-2,-2],[2,-2],[2,2],[-2,2]]),'w':obj('w','','Water',[0,-20],[[-100,-25],[100,-25],[100,-15],[-100,-15]])}}

def test_typed_road_uses_semantic_category_not_gt():
 c=TypedRoadCatalog([Entity('b','Beche Road','Building'),Entity('f','Beche Road','Footpath'),Entity('r1','Beche Road','TrafficRoad'),Entity('r2','Beche Road','TrafficRoad')]);h=c.spans('to the left of Beche Road')[0];ids,rule,conf=c.resolve(h);assert ids==['r1','r2'] and 'all_segments' in rule

def test_ambiguous_building_name_not_disambiguated():
 c=TypedRoadCatalog([Entity('a','Hall','Building'),Entity('b','Hall','Building')]);assert c.resolve(c.spans('left of Hall')[0])[0] is None

def test_unique_water_explicit_rule_has_no_target_coordinates():
 x=fixture();x['mentioned'][0].update(start=x['instruction'].index('Boat Club'),end=x['instruction'].index('Boat Club')+9);x['reference_span']=[x['mentioned'][0]['start'],x['mentioned'][0]['end']];p=generate(x);assert p['strict_ids'];q=next(c for c in p['candidates'] if c['candidate_id']==p['strict_ids'][0]);assert q['type']=='explicit_stand_face_water' and adiff(q['heading_deg'],270)<1e-8

def test_ambiguous_water_identity_abstains():
 x=fixture();x['mentioned'][0].update(start=x['instruction'].index('Boat Club'),end=x['instruction'].index('Boat Club')+9);x['scene']['w2']=copy.deepcopy(x['scene']['w']);assert not generate(x)['strict_ids']

def test_pca_keeps_all_signs_and_never_chooses_target_side():
 x=fixture();x['instruction']='The car is behind Boat Club.';x['word']='behind';x['direction_span']=[11,17];x['scene']['r']['contour']=[[-20,-2],[20,-2],[20,2],[-20,2]];x['mentioned']=[];p=generate(x);a=[c['heading_deg'] for c in p['candidates'] if c['type']=='building_PCA_axis'];assert len(a)==4 and len(set(round(v) for v in a))==4 and not p['strict_ids']

def test_no_prioritizing_most_numerous_geometry_type():
 from evaluate import candidate_weights
 p={'candidates':[{'candidate_id':str(i),'type':'PCA'} for i in range(4)]+[{'candidate_id':'road','type':'road'}]};w=candidate_weights(p,[str(i) for i in range(4)]+['road']);assert w['road']==.5 and all(w[str(i)]==.125 for i in range(4))

def test_road_direction_from_endpoint_without_target_geometry():
 # Road A extends east from road B, so explicit travel direction is +X.
 text='The house is to the left of Long Road when coming from First Street.'
 x={'occurrence_id':'x','cluster':'m:t','map':'m','instruction':text,'word':'left','direction_span':[text.index('left'),text.index('left')+4],'reference_ids':['a'],'reference_text':'Long Road','reference_span':[text.index('Long Road'),text.index('Long Road')+9],'subject_ids':['t'],'parse_tier':'high','parse_confidence':.97,'start_xy':[-50,50],'mentioned':[{'start':text.index('Long Road'),'end':text.index('Long Road')+9,'resolved_ids':['a']},{'start':text.index('First Street'),'end':text.index('First Street')+12,'resolved_ids':['b']}],'scene':{'a':obj('a','Long Road','TrafficRoad',[50,0],[[0,-2],[100,-2],[100,2],[0,2]]),'b':obj('b','First Street','TrafficRoad',[0,20],[[-2,-20],[2,-20],[2,60],[-2,60]])}}
 p=generate(x);c=[c for c in p['candidates'] if c['type']=='explicit_road_away_from_named_junction'];assert len(c)==1 and adiff(c[0]['heading_deg'],0)<2 and p['strict_ids']

def test_interior_junction_does_not_choose_best_target_side():
 text='The house is left of Long Road when coming from First Street.'
 x={'occurrence_id':'x','cluster':'m:t','map':'m','instruction':text,'word':'left','direction_span':[text.index('left'),text.index('left')+4],'reference_ids':['a'],'reference_text':'Long Road','subject_ids':['t'],'parse_tier':'high','parse_confidence':.97,'start_xy':[-10,20],'mentioned':[{'start':text.index('Long Road'),'end':text.index('Long Road')+9,'resolved_ids':['a']},{'start':text.index('First Street'),'end':text.index('First Street')+12,'resolved_ids':['b']}],'scene':{'a':obj('a','Long Road','TrafficRoad',[50,0],[[0,-2],[100,-2],[100,2],[0,2]]),'b':obj('b','First Street','TrafficRoad',[50,20],[[48,-20],[52,-20],[52,60],[48,60]])}}
 p=generate(x);assert not p['strict_ids'] and len([c for c in p['candidates'] if c['type']=='road_unsigned_axis'])==2

def test_named_region_view_does_not_become_known_when_origin_is_wide():
 x=fixture();x['instruction']='The house is to the right of Boat Club when facing it from Long Road.';text=x['instruction'];x['word']='right';x['direction_span']=[text.index('right'),text.index('right')+5];x['reference_span']=[text.index('Boat Club'),text.index('Boat Club')+9];x['scene']['road']=obj('road','Long Road','TrafficRoad',[0,20],[[-200,15],[200,15],[200,25],[-200,25]]);x['mentioned']=[{'start':text.index('Boat Club'),'end':text.index('Boat Club')+9,'resolved_ids':['r']},{'start':text.index('Long Road'),'end':text.index('Long Road')+9,'resolved_ids':['road']}];p=generate(x);c=[c for c in p['candidates'] if c['type']=='explicit_view_from_named_region'];assert c and c[0]['level']=='candidate' and not p['strict_ids']

def test_generate_ignores_private_data_if_accidentally_present():
 x=fixture();x['mentioned'][0].update(start=x['instruction'].index('Boat Club'),end=x['instruction'].index('Boat Club')+9);a=generate(x);x['GT']=[-99999,99999];x['target_position']=[500,200];x['future_trajectory']=[[0,0],[50,50]];b=generate(x);assert a==b

def test_uniform_candidate_score_is_not_any_best_hypothesis():
 from evaluate import local_score,candidate_weights
 p={'candidates':[{'candidate_id':'a','type':'axis','heading_deg':0},{'candidate_id':'b','type':'axis','heading_deg':180}]};e={'word':'left','subject_xy':[0,10],'reference_xy':[0,0],'angle_world':90};w=candidate_weights(p,['a','b']);score,err,hard=local_score(p,w,e);assert score==.5 and err==90 and not hard

def test_uniform_control_has_no_hidden_direction_choice():
 from evaluate import uniform_orientation_score
 e={'subject_xy':[0,10],'reference_xy':[0,0]};assert uniform_orientation_score(e,'sector')==.25
 assert .46<uniform_orientation_score(e,'halfplane')<.48
