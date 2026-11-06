from v5core import *
from language_v5 import role_pairs,side_scope
from geometry_v5 import projection,orientation_score

def catalog():return TypedRoadCatalog([Entity('a','Long Road','TrafficRoad'),Entity('b','First Street','TrafficRoad'),Entity('c','Hall','Building')])
def test_turn_onto_from_roles():
 t='If you turn onto Long Road from First Street this is the third building on your right.';r,h=role_pairs(t,catalog());assert r[0]['template']=='turn_onto_from';assert side_scope(t,t.index('your right')+5,r[0],h)[0]
def test_reverse_order_coming_from_onto():
 t='Coming from First Street onto Long Road this is the first house on your left.';r,h=role_pairs(t,catalog());assert r[0]['A']['resolved_ids']==['a'] and r[0]['B']['resolved_ids']==['b']
def test_enter_via():
 r,h=role_pairs('Enter Long Road via First Street, the house is to your left.',catalog());assert r[0]['template']=='enter_via'
def test_approaching_is_not_entering():
 r,h=role_pairs('Approaching Long Road from First Street the house is to your left.',catalog());assert r[0]['template'].endswith('UNDERSPECIFIED')
def test_ordinal_origin_does_not_set_travel_direction():
 r,h=role_pairs('The house on the right along Long Road. The sixth from First Street.',catalog());assert r[0]['template'].endswith('UNDERSPECIFIED')
def test_anonymous_alley_not_road_reference():
 t='The second building on the right from the first alley that is on Long Road when coming from First Street.';r,h=role_pairs(t,catalog());ok,reason,pop=side_scope(t,t.index('right'),r[0],h);assert not ok and not pop
def test_alley_after_turn_resets_reference_frame():
 t='If you turn onto Long Road from First Street, then take a right into an alley, this is the second building on your left.';r,h=role_pairs(t,catalog());assert not side_scope(t,t.index('your left')+5,r[0],h)[0]
def test_named_hall_not_replaced_by_travel_road():
 t='If you turn onto Long Road from First Street this is the building to the left of Hall.';r,h=role_pairs(t,catalog());assert not side_scope(t,t.index('left'),r[0],h)[0]
def test_lane_and_turn_subregions_remain_unknown():
 t='The house on the right side lane along Long Road from First Street.';r,h=role_pairs(t,catalog());assert not side_scope(t,t.index('right'),r[0],h)[0]
def test_observer_switch_not_inherited_from_earlier_road_frame():
 t='If you turn onto Long Road from First Street, then stand facing Hall, this is the building to your left.';r,h=role_pairs(t,catalog());assert not side_scope(t,t.index('your left')+5,r[0],h)[0]
def test_common_predicate_and_no_best_PCA_sign():
 f={'points_xy':[[0,0],[5,0],[10,0]],'edges':[[0,1],[1,2]],'heading_deg':[0,0,0]};p=projection(f,[[3,4],[4,4],[4,6],[3,6]]);a=orientation_score(p,'left');b=orientation_score(p,'left',heading=0);rev=orientation_score(p,'left',rotation=180);assert a['result']==b['result']=='consistent' and rev['result']=='conflict';assert (a['score']+rev['score'])/2==.5
def test_mask_is_shared_by_every_orientation():
 f={'points_xy':[[0,0],[5,0],[10,0]],'edges':[[0,1],[1,2]],'heading_deg':[0,0,0]};p=projection(f,[[3,-2],[4,-2],[4,2],[3,2]]);assert all(orientation_score(p,'left',heading=h)['result']=='ambiguous' for h in [0,90,180,270])
def test_zero_length_smoothing_edges_do_not_create_nan_or_change_side():
 f={'points_xy':[[0,0],[5,0],[5,0],[10,0]],'edges':[[0,1],[1,2],[2,3]],'heading_deg':[0,0,0,0]};p=projection(f,[[3,4],[4,4],[4,6],[3,6]]);assert orientation_score(p,'left')['result']=='consistent'
