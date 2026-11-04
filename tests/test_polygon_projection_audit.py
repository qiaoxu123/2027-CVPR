import numpy as np
from affine import Affine
from graphnav.polygon_projection_audit import audit_projection,raster_projection_discrepancy


def test_large_landmark_can_cover_view_but_fail_full_object_visibility_cutoff():
    footprint=np.array([[0,0],[10,0],[10,10],[0,10]],float)
    poly=np.array([[-20,-20],[30,-20],[30,30],[-20,30]],float)
    result=audit_projection(poly,footprint,np.ones((384,384,3),np.uint8),np.ones((384,384),bool))
    assert abs(result['geometry']['polygon_coverage']-.04)<1e-6
    assert result['geometry']['view_occupancy']==1
    assert result['mask'].all()
    assert not result['thresholds']['roi50_256'] and not result['thresholds']['roi90_256']


def test_outside_has_empty_mask_and_partial_contour_is_clipped():
    footprint=np.array([[0,0],[10,0],[10,10],[0,10]],float)
    rgb=np.ones((384,384,3),np.uint8);valid=np.ones((384,384),bool)
    outside=audit_projection(np.array([[20,20],[30,20],[30,30],[20,30]]),footprint,rgb,valid)
    assert outside['geometry']['polygon_coverage']==0 and outside['mask'].sum()==0
    partial=audit_projection(np.array([[5,0],[15,0],[15,10],[5,10]]),footprint,rgb,valid)
    assert abs(partial['geometry']['polygon_coverage']-.5)<1e-6
    assert 0<partial['mask'].sum()<384*384
    assert partial['roundtrip_max_px']<1e-3


def test_invalid_observed_pixels_do_not_become_semantic_visibility():
    footprint=np.array([[0,0],[10,0],[10,10],[0,10]],float)
    rgb=np.zeros((384,384,3),np.uint8)
    result=audit_projection(footprint,footprint,rgb,np.zeros((384,384),bool))
    assert result['mask'].all() and not result['source_valid_mask'].any()
    assert result['actual_visibility']=='unknown'


def test_raster_rounding_is_separate_from_analytic_roundtrip():
    fp=np.array([[.3,10.4],[10.3,10.4],[10.3,.4],[.3,.4]],float)
    transform=Affine(1,0,0,0,-1,20)
    result=raster_projection_discrepancy(fp,transform)
    assert result['max_center_convention_px']>1
    exact=np.array([[.5,10.5],[10.5,10.5],[10.5,.5],[.5,.5]])
    assert raster_projection_discrepancy(exact,transform)['max_center_convention_px']<1e-3


def test_rotated_footprint_changes_pixel_orientation_without_world_scale_error():
    footprint=np.array([[0,0],[10,0],[10,10],[0,10]],float)
    poly=np.array([[2,2],[4,2],[4,8],[2,8]],float)
    rgb=np.ones((384,384,3),np.uint8);valid=np.ones((384,384),bool)
    normal=audit_projection(poly,footprint,rgb,valid)
    rotated=audit_projection(poly,np.roll(footprint,1,axis=0),rgb,valid)
    assert np.array_equal(rotated['mask'],np.rot90(normal['mask'],-1))
    assert normal['geometry']==rotated['geometry']
    diamond=np.array([[0,5],[5,0],[10,5],[5,10]],float)
    outside=audit_projection(np.array([[0,0],[1,0],[1,1],[0,1]]),diamond,rgb,valid)
    assert outside['geometry']['polygon_coverage']==0 and not outside['mask'].any()
