from types import SimpleNamespace
import numpy as np
from scripts.audit_v2_candidate_observability import score

class Renderer:
    def current_rgbd(self,*args):
        return dict(bgr=np.full((64,64,3),128,np.uint8),depth_valid=np.ones((64,64),bool),footprint_world=np.array([[0.,0.],[64.,0.],[64.,64.],[0.,64.]]))

def test_geometry_pixel_support_is_only_a_proxy():
    ep=SimpleNamespace(map_name='example',target_position=np.array([20.,20.,0.]))
    r=score(ep,[[0,0,50,0]],dict(contour=[[10,10],[30,10],[30,30],[10,30]]),Renderer(),0)[0]
    assert r['center_in_footprint'] and r['valid_roi'] and r['clarity_proxy_16']
    assert not r['clarity_proxy_32']
    assert 'visible' not in r and 'recognized' not in r

def test_missing_contour_and_disagreement_stay_unknown():
    ep=SimpleNamespace(map_name='example',target_position=np.array([20.,20.,0.]))
    r=score(ep,[[0,0,50,0]],{},Renderer(),0)[0]
    assert r['center_in_footprint'] and not r['valid_roi'] and not r['polygon_available']
    ep.target_position=np.array([150.,150.,0.])
    r=score(ep,[[0,0,50,0]],dict(contour=[[10,10],[30,10],[30,30],[10,30]]),Renderer(),0)[0]
    assert r['geometric_overlap'] and r['center_contour_disagreement_gt20m']
    assert not r['valid_roi'] and not r['clarity_proxy_8']
