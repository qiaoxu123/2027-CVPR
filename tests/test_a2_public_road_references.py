import copy

import pytest

from experiments.a2_waldo_landmark_belief.public_road_references import resolve_road_references
from experiments.a2_waldo_landmark_belief.public_regions import region_shape


def obj(i, x=0, y=0, kind='TrafficRoad', name='Main Road'):
    return dict(id=i, name=name, object_type=kind,
                contour=[[x,y],[x+1,y],[x+1,y+1],[x,y+1]])


def test_touching_road_footpath_becomes_one_literal_public_reference():
    objects = [obj(1), obj(2, x=1, kind='Footpath')]
    before = copy.deepcopy(objects)
    refs = resolve_road_references('map', ['Main Road', ' main road '], objects)
    assert len(refs) == 1 and refs[0]['linked']
    assert refs[0]['member_ids'] == [1, 2]
    assert region_shape(refs[0]['region']).area == pytest.approx(2)
    assert refs[0]['height_policy'] == 'missing_for_public_road_group'
    assert refs[0]['name'] == 'Main Road'
    assert objects == before


@pytest.mark.parametrize('objects', [
    [obj(1),obj(2,x=2)],
    [obj(1,kind='Building'),obj(2,x=1,kind='Building')],
    [obj(1,kind='Footpath'),obj(2,x=1,kind='Footpath')],
    [obj(1),obj(2,x=1,kind='Building')],
])
def test_unsupported_homonyms_remain_unlinked(objects):
    refs = resolve_road_references('map', ['Main Road'], objects)
    assert len(refs) == 2 and not any(r['linked'] for r in refs)


def test_one_disconnected_member_blocks_the_entire_name_group():
    refs = resolve_road_references('map',['Main Road'],[obj(1),obj(2,x=1),obj(3,x=10)])
    assert len(refs) == 3 and not any(r['linked'] for r in refs)


def test_point_touch_group_keeps_multiple_components():
    refs = resolve_road_references('map',['Main Road'],[obj(1),obj(2,x=1,y=1)])
    assert len(refs) == 1 and refs[0]['linked']
    assert region_shape(refs[0]['region']).geom_type == 'MultiPolygon'


def test_invalid_member_does_not_authorize_partial_group():
    bad = obj(2,x=1)
    bad['contour'] = [[0,0],[float('nan'),0],[1,1]]
    refs = resolve_road_references('map',['Main Road'],[obj(1),bad])
    assert len(refs) == 2 and not any(r['linked'] for r in refs)


def test_singleton_and_unresolved_follow_legacy_behavior():
    from experiments.a2_waldo_landmark_belief.data_protocol import resolve_public_references
    objects = [obj(1,kind='Building')]
    assert resolve_road_references('map',['Main Road','Absent'],objects) == resolve_public_references('map',['Main Road','Absent'],objects)


def test_input_and_query_permutations_preserve_entity_and_geometry():
    objects = [obj(8),obj(3,x=1,kind='Footpath')]
    left = resolve_road_references('map',['Main Road',' main road '],objects)
    right = resolve_road_references('map',[' main road ','Main Road'],objects[::-1])
    assert left == right


def test_exact_duplicate_public_ids_deduplicate_but_conflicting_geometry_fails():
    first = obj(1)
    assert len(resolve_road_references('map',['Main Road'],[first,copy.deepcopy(first)])) == 1
    with pytest.raises(ValueError,match='Conflicting public object'):
        resolve_road_references('map',['Main Road'],[first,obj(1,x=10)])


def test_fuzzy_name_is_not_authorized():
    refs = resolve_road_references('map',['Main Rd'],[obj(1),obj(2,x=1)])
    assert len(refs) == 1 and not refs[0]['linked']


def test_group_and_singleton_names_do_not_collide_by_substring():
    refs = resolve_road_references('map',['Main Road','Main'],[obj(1),obj(2,x=1),obj(3,name='Main')])
    # Text-overlap ambiguity is still handled by the unchanged mention matcher.
    assert len(refs) == 2 and all(r['linked'] for r in refs)


def test_new_group_overlapping_an_existing_mention_keeps_both_unconfirmed():
    from experiments.a2_waldo_landmark_belief.mentions import landmark_mention_spans
    from experiments.a2_waldo_landmark_belief.data_protocol import resolve_public_references
    objects = [obj(1),obj(2,x=1),obj(3,name='Main')]
    queries = ['Main Road', 'Main']
    offsets = [[0,4],[5,9]]
    old = resolve_public_references('map',queries,objects)
    new = resolve_road_references('map',queries,objects)
    assert landmark_mention_spans('Main Road',offsets,old)[1].sum() == 1
    assert not landmark_mention_spans('Main Road',offsets,new)[1].any()
