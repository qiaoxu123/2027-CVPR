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


def test_persistent_pixel_vectors_do_not_reencode_or_retain_temporary_rgb(tmp_path):
    from collections import OrderedDict
    from types import SimpleNamespace
    import torch
    from graphnav.observed_landmarks import ObservedLandmarkEncoder
    encoder=ObservedLandmarkEncoder.__new__(ObservedLandmarkEncoder)
    encoder.device=torch.device('cpu');encoder.directory=tmp_path;encoder.vector_directory=tmp_path/'vectors';encoder.vector_directory.mkdir()
    encoder.vectors=OrderedDict();encoder.cache_size=4;encoder.last_step=-1;encoder.history=[]
    class Backbone:
        calls=0
        def encode(self,paths):
            self.calls+=1;return torch.ones(len(paths),4,8),torch.ones(len(paths),4,dtype=torch.bool)
    class Head:
        query=torch.ones(1,1,6)
        def __call__(self,patches,mask):return torch.ones(len(patches),6)*.25
    encoder.backbone=Backbone();encoder.head=Head()
    _,first=encoder.observe(observation(),[landmark()],step=0)
    encoder.vectors.clear();_,second=encoder.observe(observation(),[landmark()],step=1)
    assert torch.equal(first,second) and encoder.backbone.calls==1
    assert not list(tmp_path.glob('*.png')) and len(list(encoder.vector_directory.glob('*.pt')))==1
    encoder.start_episode();assert encoder.last_step==-1 and encoder.history==[]
