import numpy as np
from graphnav.observed_landmarks import current_landmark_rois


def observation():
    return dict(bgr=np.full((384,384,3),120,dtype=np.uint8),depth_valid=np.ones((384,384),dtype=bool),
        footprint_world=np.array([[0,100],[100,100],[100,0],[0,0]]),pose=np.array([50,50,100,0]))


def landmark():
    return dict(id='landmark',name='library',object_type='Building',contour=[[20,20],[40,20],[40,40],[20,40]])


def test_all_visible_map_landmarks_do_not_require_a_target_or_candidate_list():
    rois=current_landmark_rois(observation(),[landmark()])
    assert len(rois)==1 and rois[0].name=='library'
    assert 0<rois[0].confidence<1 and rois[0].roi_image.shape==(384,384,3)
    assert rois[0].quality['semantic_agreement']=='unverified geometric proposal'


def test_missing_source_is_unknown_and_depth_uncertainty_reduces_confidence():
    source=observation();normal=current_landmark_rois(source,[landmark()])[0]
    source['depth_valid'][:]=False;weak=current_landmark_rois(source,[landmark()])[0]
    assert weak.confidence<normal.confidence
    source['bgr'][:]=0;assert current_landmark_rois(source,[landmark()])==[]
