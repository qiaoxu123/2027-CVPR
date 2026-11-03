from scripts.audit_public_reference_name_sets import public_group


def test_homonymous_roads_remain_separate_and_ignore_private_position():
    objects = {str(i): dict(name='Main Road', object_type='TrafficRoad',
        contour=[[i,0],[i+1,0],[i+1,1],[i,1]], position=[999*i,999*i]) for i in (1,2)}
    mention = dict(public_ids=['2','1'],public_names=['Main Road'])
    first = public_group(mention,objects)
    objects['1']['position'] = [-99999,0]
    assert public_group(mention,objects) == first
    assert first['public_ids'] == first['exact_binding_ids'] == ['1','2']
    assert first['kind'] == 'homonymous_road_polygons'


def test_mixed_type_name_is_uncertain_alternatives():
    objects = {str(i):dict(name='The Green',object_type=t,
        contour=[[0,0],[1,0],[1,1],[0,1]]) for i,t in enumerate(['Building','Ground'])}
    group = public_group(dict(public_ids=['0','1'],public_names=['The Green']),objects)
    assert group['kind'] == 'name_or_type_alternatives'
    assert len(group['public_ids']) == 2
