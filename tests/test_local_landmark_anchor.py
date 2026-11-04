import numpy as np
from graphnav.local_landmark_anchor import local_landmark_anchor,named_public_polygons,local_anchor_tokens


def test_large_polygon_has_local_anchor_without_whole_object_fraction_gate():
    fp=np.array([[0,10],[10,10],[10,0],[0,0]],float)
    poly=np.array([[-20,-20],[30,-20],[30,30],[-20,30]],float)
    obs=dict(bgr=np.ones((384,384,3),np.uint8),depth_valid=np.ones((384,384),bool),footprint_world=fp)
    a=local_landmark_anchor([poly],obs)
    assert a['whole_fraction']<.05 and a['view_fraction']==1
    assert a['pixel_usable'] and a['train_eligible']
    assert (a['labels']==1).all() and not (a['labels']==0).any()
    assert a['valid_mask'].all()


def test_all_named_road_pieces_are_checked_not_just_first_outside_piece():
    objects={'a':dict(name='Road X',object_type='TrafficRoad',contour=[[20,20],[25,20],[25,25],[20,25]]),
             'b':dict(name='Road X',object_type='Footpath',contour=[[1,1],[5,1],[5,5],[1,5]]),
             'c':dict(name='Other',object_type='Building',contour=[[0,0],[10,0],[10,10],[0,10]])}
    result=named_public_polygons(objects,'Road X',expected_type='Road')
    assert result['ids']==['a','b']
    obs=dict(bgr=np.ones((384,384,3),np.uint8),depth_valid=np.ones((384,384),bool),footprint_world=np.array([[0,10],[10,10],[10,0],[0,0]]))
    a=local_landmark_anchor(result['contours'],obs)
    assert a['pixel_usable'] and a['geometry_mask'].any()


def test_invalid_pixels_do_not_become_evidence_and_partial_valid_regions_remain():
    fp=np.array([[0,10],[10,10],[10,0],[0,0]],float)
    obs=dict(bgr=np.ones((384,384,3),np.uint8),depth_valid=np.zeros((384,384),bool),footprint_world=fp)
    absent=local_landmark_anchor([fp],obs)
    assert not absent['pixel_usable'] and not absent['train_eligible']
    assert (absent['labels']==-1).all()
    obs['depth_valid'][:,:192]=True
    partial=local_landmark_anchor([fp],obs)
    assert partial['pixel_usable'] and partial['train_eligible']
    assert partial['valid_mask'][40:340,40:190].all() and not partial['valid_mask'][:,192:].any()
    assert not partial['valid_mask'][0].any()  # true Polygon boundary stays uncertain


def test_local_tokens_keep_registered_position_and_clip_to_actual_view():
    left=np.array([[.5,1],[3,1],[3,9],[.5,9]],float)
    right=left+[6,0]
    fp=np.array([[0,10],[10,10],[10,0],[0,0]],float)
    a=local_anchor_tokens([left],fp);b=local_anchor_tokens([right],fp)
    assert a.shape==(16,4) and b.shape==(16,4)
    assert b[:,0].mean()>a[:,0].mean()+.5
    assert (a[:,:2]>=0).all() and (a[:,:2]<=1).all()


def test_fast_native_patch_validity_is_bit_identical_to_old_dense_roi_path():
    from graphnav.local_landmark_anchor import native_patch_validity
    from graphnav.native_dense_regions import native_dense_regions,sparse_view
    rng=np.random.default_rng(0);rgb=np.full((384,384,3),100,np.uint8);valid=rng.random((384,384))>.04
    valid[:100,:120]=False;rgb[200:220,180:210]=0
    obs=dict(bgr=rgb,depth_valid=valid,footprint_world=np.array([[0,10],[10,10],[10,0],[0,0]]))
    assert np.array_equal(native_patch_validity(rgb,valid),sparse_view(native_dense_regions(obs))['valid'])
