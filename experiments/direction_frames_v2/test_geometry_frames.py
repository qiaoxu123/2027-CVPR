import numpy as np
from shapely.geometry import Polygon
from geometry_frames import transforms,shape_axis,initial_motion,approach_motion,boundary_class,angle,evaluate

def test_d4_rotations_and_mirrors_include_exactly_eight_distinct_maps():
 ts=transforms();assert len(ts)==12
 assert len({tuple(np.array(t['matrix']).ravel()) for t in ts})==8
 for t in ts:
  m=np.array(t['matrix']);assert np.allclose(m.T@m,np.eye(2)) and abs(round(np.linalg.det(m)))==1

def test_pca_is_unsigned_and_density_independent():
 p=Polygon([(0,0),(20,0),(20,2),(0,2)])
 a=shape_axis(p);assert a['reliable_unsigned_axis'] and not a['directed_heading_verified']
 assert min(a['axis_deg'],180-a['axis_deg'])<2

def test_pca_circle_square_abstains():
 assert not shape_axis(Polygon([(0,0),(2,0),(2,2),(0,2)]))['reliable_unsigned_axis']

def test_stationary_initial_turns_do_not_produce_motion_heading():
 h,w=initial_motion([[0,0,1,1,0,0],[0,0,1,0,1,0],[0,4,1,0,1,0]])
 assert h==90 and w==[0,2]

def test_approach_at_start_is_unavailable():
 assert approach_motion([[0,0],[5,5]],0)==(None,None)

def test_incoming_motion_ignores_stationary_last_segment():
 h,w=approach_motion([[0,0],[5,0],[5,0]],2);assert h==0 and w==[0,2]

def test_complete_boundaries_can_be_ambiguous_even_disjoint_centers():
 t=Polygon([(4,2),(6,2),(6,4),(4,4)]);r=Polygon([(0,0),(10,0),(10,1),(0,1)])
 cl,g,_=boundary_class(t,r,np.array([1,0]));assert cl=='ambiguous' and g==-6

def test_signed_axis_hypotheses_never_merge_into_best_per_target():
 pair={'semantic_word':'left','angle_world':90.,'dx':0,'dy':10,'occurrence_id':'x','cluster':'x','map':'m','confidence_tier':'high','parse_confidence':.97,'geometry_status':'disjoint','reference_category':'TrafficRoad','target_category':'Car','relation_subject_category':'Car','explicit_view':False,'local_layout':True}
 a=evaluate(pair,None,None,'plus',{'heading_deg':0});b=evaluate(pair,None,None,'minus',{'heading_deg':180})
 assert a['sector_correct'] and not b['sector_correct']
