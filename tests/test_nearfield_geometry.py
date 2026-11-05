from shapely.geometry import box
from scripts.nearfield_geometry import legal_polygon, neighbors, relation, road_enclosures


def test_all_neighbors_no_twelve_limit_and_boundary_not_centroid():
    objects = {str(i): {'contour': list(box(i, 0, i+1, 1).exterior.coords)} for i in range(20)}
    objects['long'] = {'contour': list(box(5, 5, 1000, 6).exterior.coords)}
    assert len(neighbors(objects, box(0, 0, 1, 1), set(), 25)) == 21


def test_containment_and_direction_are_not_instance_or_visual_labels():
    r = relation(box(0, 0, 20, 20), box(12, 12, 13, 13))
    assert r['first_covers_second'] and r['boundary_distance_m'] == 0
    assert r['delta_centroid_world_xy'] == [2.5, 2.5]
    assert r['evidence'] == 'public_polygon_geometry_only'


def test_complete_enclosure_and_missing_road_unknown():
    roads = [('a', box(0, 0, 10, 1)), ('b', box(0, 9, 10, 10)),
             ('c', box(0, 0, 1, 10)), ('d', box(9, 0, 10, 10))]
    holes = road_enclosures(roads)
    assert len(holes) == 1 and holes[0]['area_m2'] == 64
    assert len(holes[0]['adjacent_road_ids']) == 4
    assert road_enclosures(roads[:-1]) == []


def test_invalid_polygon_rejected_without_silent_geometry_repair():
    assert legal_polygon([[0, 0], [1, 1], [0, 1], [1, 0]]) is None
    assert legal_polygon([[0, 0], [float('nan'), 1], [1, 0]]) is None


def test_normalized_road_label_stays_road_for_anchor_proxy_scoring():
    from scripts.build_nearfield_scene_dataset import GROUP
    assert GROUP['Road'] == GROUP['TrafficRoad'] == 'Road'
    assert GROUP['Parking'] == 'Parking'


def test_multi_polygon_road_is_not_multiple_distinct_roads():
    from scripts.nearfield_geometry import enclosure_road_names
    same = {str(i): dict(name='Ring Road') for i in range(4)}
    assert enclosure_road_names(list(same), same)['distinct_named_road_count'] == 1
    distinct = {str(i): dict(name=f'Road {i}') for i in range(4)}
    assert enclosure_road_names(list(distinct), distinct)['distinct_named_road_count'] == 4
    assert enclosure_road_names(['missing'], {})['distinct_named_road_count'] == 0
