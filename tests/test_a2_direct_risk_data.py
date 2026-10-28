import numpy as np
import pytest
from experiments.a2_waldo_landmark_belief.direct_risk_data import validate_arrays,normalization


def test_outcome_arrays_must_agree_with_actual_distances():
    d=dict(features=np.zeros((3,15),np.float32),before_ne=np.array([40.,10.,10.]),
           after_ne=np.array([[10.]*4,[40.]*4,[10.]*4]),outcomes=np.array([[1]*4,[-1]*4,[0]*4],np.int8))
    ids=[dict(map_name='m',object_id=i,episode=i,step=0) for i in range(3)]
    validate_arrays(d,ids,3)
    d['outcomes'][0,3]=0
    with pytest.raises(ValueError):validate_arrays(d,ids,3)


def test_only_training_rows_determine_normalization_and_class_weights():
    d=dict(features=np.arange(90,dtype=np.float32).reshape(6,15),outcomes=np.array([[0]*4,[0]*4,[1]*4,[-1]*4,[1]*4,[0]*4]))
    mean,std,weights=normalization(d)
    assert np.allclose(mean,d['features'].mean(0)) and np.allclose(std,d['features'].std(0))
    assert np.allclose(weights,[6/9,1.,2.])
