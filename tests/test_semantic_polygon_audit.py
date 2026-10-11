import numpy as np

from experiments.semantic_polygon_grounding.audit import (
    anonymous_reference_flags, geometry_summary, goal_metrics, inspect_query,
    name_binding, valid_polygon,
)
from graphnav.landmark_alias import build_name_index


def objects():
    return {
        '1': dict(name='White Church',object_type='Building',contour=[[0,0],[5,0],[5,5],[0,5]]),
        '2': dict(name='Hall',object_type='Building',contour=[[10,0],[15,0],[15,5],[10,5]]),
        '3': dict(name='Hall',object_type='Building',contour=[[20,0],[25,0],[25,5],[20,5]]),
    }


def test_homonyms_preserved_no_fake_instance_truth():
    o=objects(); b=name_binding('Hall',o,build_name_index(o))
    assert b['candidate_ids']==['2','3']
    assert b['reference_instance_truth'] is None
    assert not b['unique_valid']


def test_private_target_coordinates_cannot_choose_reference():
    o=objects(); idx=build_name_index(o)
    a=name_binding('Hall',o,idx)
    for obj in o.values(): obj['target_world']=[9999,9999];obj['target_id']='2'
    assert name_binding('Hall',o,idx)==a


def test_named_reference_not_counted_as_anonymous():
    o=objects(); q=inspect_query('A house beside the White Church', ['White Church'],'m',{'m':o},{'m':build_name_index(o)})
    assert q['all_named_bindings_unique_valid']
    assert q['anonymous_reference_phrases']==[]


def test_unnamed_reference_flag_is_not_semantic_truth():
    assert anonymous_reference_flags('a house between the white church and a black building')==['white church']
    o=objects();q=inspect_query('house near the gray building',[],'m',{'m':o},{'m':build_name_index(o)})
    assert q['anonymous_reference_phrases']==['gray building']
    assert not q['identifiable_ambiguous_anchor_truth']


def test_invalid_polygon_not_unique_valid_binding():
    o={'1':dict(name='Invalid',object_type='Building',contour=[[0,0],[2,2],[0,2],[2,0]])}
    assert valid_polygon(o['1']) is None
    assert not name_binding('Invalid',o,build_name_index(o))['unique_valid']


def test_goal_metrics_keep_grid_radius_and_ties():
    metrics,ranking,rank=goal_metrics([1,1,0],[[0,0],[.5,0],[1,0]],[0,0,100,100],[50,0])
    assert ranking==[1,0,2]
    assert metrics==dict(goal_r1=1.,goal_r4=1.,goal_r8=1.,ne=0.)
    assert rank==1


def test_geometry_uses_polygon_not_target_position():
    o=objects()['1'];a=geometry_summary(o);o['position']=[1000,1000,1000]
    assert geometry_summary(o)==a
    assert np.isclose(a['area_m2'],25)
