"""Adversarial input fixtures for scope/geometry; synthetic data ONLY in tests."""
import importlib.util
from pathlib import Path
import numpy as np
import pytest
spec=importlib.util.spec_from_file_location('analysis',Path(__file__).with_name('analyze.py'));a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)
OBJ={'1':{'name':'Church','position':[0,0,0]},'2':{'name':'West Midlands Police Custody Suite','position':[50,50,0]}}
def parse(text,word='behind',obj=None,cat='Car'):
 m=next(m for m in a.TOKEN.finditer(text) if m[0].lower()==word)
 return a.parse(text,m,obj or OBJ,cat)

def test_simple_direct_relation():
 r=parse('The white car behind the Church.')
 assert r['scope']=='direct' and r['ref_id']=='1'

def test_front_relation_is_not_part():
 r=parse('The white car in front of the Church.','front')
 assert r['scope']=='direct' and r['ref_id']=='1'

def test_indirect_target_parking_church_not_flattened():
 r=parse('The white car in the parking lot behind the Church.')
 assert r['scope']=='indirect' and r['status']=='excluded'

def test_indirect_reference_yard_not_flattened():
 r=parse('The white car behind the yard of the Church.')
 assert r['scope']=='indirect'

def test_anonymous_car_cannot_be_replaced_by_nearby_church():
 r=parse('The white car behind a red car near the Church.')
 assert r['status']=='excluded'

def test_repeated_name_never_disambiguated_by_target_gt():
 o={'1':{'name':'Church'},'2':{'name':'Church'}}
 r=parse('The white car behind the Church.',obj=o)
 assert r['reason']=='duplicate_landmark_name'

def test_proper_name_direction():
 r=parse('The white car beside the West Midlands Police Custody Suite.','west')
 assert r['reason']=='direction_in_proper_name'

def test_right_next_to_is_emphasis():
 assert parse('The white car right next to the Church.','right')['reason']=='right_emphasis'

def test_front_yard_is_part():
 assert parse('The white car has a front bumper.','front')['status']=='excluded'

def test_nested_subject_not_target():
 assert parse('The white car with a black car behind the Church.')['scope']=='indirect'

def test_preceding_relation_scope_uncertain():
 assert parse('The white car below another car behind the Church.')['status']=='excluded'

def test_diagonal_not_single_left():
 assert parse('The white car to the upper left of the Church.','left')['status']=='excluded'

def test_subregion_not_named_whole():
 assert parse('The white car behind the Church car park.')['scope']=='indirect'

def test_wrap_and_sector():
 assert a.distance(359,1)==2
 assert a.angle(0,-1)==270

def obj(x0,y0,x1,y1):
 return {'contour':[[x0,y0],[x1,y0],[x1,y1],[x0,y1]],'position':[(x0+x1)/2,(y0+y1)/2,0]}

def test_entire_boundary_not_centroid():
 t=obj(9,-1,12,1);r=obj(0,-10,10,10)
 g=a.geometry(t,r,[10.5,0,0])
 assert g['dx']>0 and g['gap_0']<0 and g['geometry_status'].startswith('ambiguous')

def test_disjoint_boundary_projection():
 t=obj(12,-1,14,1);r=obj(0,-10,10,10)
 g=a.geometry(t,r,[13,0,0]);assert g['gap_0']==2 and g['geometry_status']=='valid'

def test_invalid_polygon_not_repaired():
 t={'contour':[[0,0],[1,1],[0,1],[1,0]],'position':[.5,.5,0]}
 assert a.geometry(t,obj(5,5,6,6),[.5,.5,0])['geometry_status']=='missing_or_invalid_polygon'

def test_wilson_small_n():
 lo,hi=a.wilson(1,1);assert .2<lo<.21 and hi==1

def test_enclosed_ground_does_not_make_buildings_relation_direct():
 assert parse("The green lawn enclosed by the buildings behind the Church.",cat='Ground')['status']=='excluded'

def test_nearest_trees_scope_abstains():
 assert parse('A tall white building nearest to the trees behind the Church.',cat='Building')['status']=='excluded'
