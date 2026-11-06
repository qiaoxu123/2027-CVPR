"""Mechanism and scope tests, not invented experimental outcomes."""
import copy
import numpy as np
from shared import *
from shapely.geometry import box
from frames import derive,angular_support
from controls import role_view
from road_language import extract
from roads import directed_field

def object_(i,name,category,p):return {'id':i,'name':name,'category':category,'position':list(p.centroid.coords[0]),'contour':list(p.exterior.coords)}
def sample():
 scene={'a':object_('a','Long Road','TrafficRoad',box(0,-3,120,3)),'b':object_('b','First Street','TrafficRoad',box(-3,-30,3,30)),'c':object_('c','Hall','Building',box(25,20,30,25))}
 text='If you turn onto Long Road coming from First Street this is the first building to your left.'
 return {'occurrence_id':'test','map':'m','cluster':'m:t','subject_ids':['t'],'subject_role':'annotated_target','instruction':text,'reference_ids':['a'],'reference_span':[text.index('Long Road'),text.index('Long Road')+9],'word':'left','direction_span':[text.index('your left')+5,text.index('your left')+9],'start_xy':[0,-20],'mentioned':[{'start':text.index(o['name']),'end':text.index(o['name'])+len(o['name']),'resolved_ids':[i]} for i,o in scene.items() if o['name'] in text],'scene':scene}

def test_direct_travel_side_extraction():
 cat=TypedRoadCatalog([Entity('a','Long Road','TrafficRoad'),Entity('b','First Street','TrafficRoad')]);r=extract(sample()['instruction'],cat);assert len(r)==1 and r[0]['reference_ids']==['a'] and r[0]['word']=='left'

def test_alley_target_not_simplified_to_road_side():
 cat=TypedRoadCatalog([Entity('a','Long Road','TrafficRoad'),Entity('b','First Street','TrafficRoad')]);assert not extract('If you turn onto Long Road coming from First Street this is the first car in the alley to your left.',cat)

def test_road_positive_direction_uses_independent_endpoint():
 x=sample();f=directed_field(x,['a'],['b']);assert f['state']=='unique_field' and abs(np.median(f['heading_deg']))<10

def test_internal_entry_keeps_two_directions():
 x=sample();x['scene']['b']=object_('b','First Street','TrafficRoad',box(57,-30,63,30));f=directed_field(x,['a'],['b']);assert f['state']!='unique_field'

def test_wide_observer_region_is_not_a_point():
 s=angular_support(box(-100,-2,100,2),box(-2,20,2,25));assert s['width_deg']>100

def test_region_direction_reverses_when_roles_swap():
 a,b=box(-2,-2,2,2),box(98,-2,102,2);u,v=angular_support(a,b),angular_support(b,a);assert abs(float(adiff(u['heading_deg'],v['heading_deg']))-180)<1e-7 and abs(u['width_deg']-v['width_deg'])<1e-7
 c=box(-2,98,2,102);w=angular_support(a,c);assert abs(float(adiff(u['heading_deg'],w['heading_deg']))-90)<1e-7

def test_relation_scopes_do_not_inherit_later_observer():
 x=sample();t='The house is behind Hall. If you stand at Long Road facing First Street, the house is to the right of Hall.';x.update(instruction=t,reference_ids=['c'],reference_span=[t.index('Hall'),t.index('Hall')+4],word='behind',direction_span=[t.index('behind'),t.index('behind')+6]);x['mentioned']=[{'start':m.start(),'end':m.end(),'resolved_ids':[i]} for i,o in x['scene'].items() for m in re.finditer(re.escape(o['name']),t)];p=derive(x,{});assert p['frame_type']=='intrinsic' and p['state']!='unique'
 y=sample();y['scene']['a']['name']='The Anchor';y['scene']['b']['name']='W. D. M. Cars';z=derive(role_view(y,['a'],['b']),{});assert z['frame_type']=='observer' and z['origin_region_ids']==['a'] and z['state']!='unknown'

def test_missing_viewpoint_stays_unknown():
 x=sample();t='The house is right of Hall when facing First Street.';x.update(instruction=t,reference_ids=['c'],reference_span=[t.index('Hall'),t.index('Hall')+4],word='right',direction_span=[t.index('right'),t.index('right')+5]);x['mentioned']=[{'start':m.start(),'end':m.end(),'resolved_ids':[i]} for i,o in x['scene'].items() for m in re.finditer(re.escape(o['name']),t)];p=derive(x,{});assert p['state']=='unknown' and p['forward_heading_deg'] is None

def test_private_GT_and_future_fields_are_ignored():
 x=sample();cache={};a=derive(x,cache);x['target_GT']=[-9999,8888];x['future_trajectory']=[[0,0],[99,88]];assert derive(x,cache)==a

def test_full_contour_and_sign_reversal():
 from evaluate import field_geometry
 x=sample();f=directed_field(x,['a'],['b']);contour=list(box(30,8,35,15).exterior.coords);assert field_geometry(f,contour,'left')['result']=='consistent';assert field_geometry(f,contour,'right',rotation=180)['result']=='consistent';assert field_geometry(f,contour,'right')['result']=='conflict'
