from copy import deepcopy
from scripts.cache_local_chain_training import public_query


def test_private_labels_cannot_change_frozen_prior_query():
    shared=dict(map_name='map',instruction='The red car near Station.',
                world_bounds=[0,0,400,400],physical_map_sizes=[400,400],
                landmarks=[dict(name='Station',object_type='Building',aliases=[],contour=[[.1,.1],[.2,.1],[.2,.2]])],
                dataset_version='revised',match_mode='fuzzy')
    first=deepcopy(shared); first.update(target_xy=[.8,.8],target_position=[320,320],target_id=4,labels={'box':[1,2,3,4]})
    second=deepcopy(shared);second.update(target_xy=[.1,.9],target_position=[40,360],target_id=77,labels={'box':[9,7,5,3]})
    assert public_query(first)==public_query(second)
    assert public_query(first)['target_xy']==[0.,0.]
    assert not {'target_position','target_id','labels'} & set(public_query(first))
