import numpy as np
from graphnav.observed_landmarks import LandmarkSpatialIndex


def test_index_matches_scalar_inclusive_filter_without_reordering_or_changing_inputs():
    rng=np.random.default_rng(3);low=rng.uniform(-100,100,(500,2));high=low+rng.uniform(0,20,(500,2))
    low=np.vstack([low,[-10,-10],[10,10]]).astype(np.float32)
    high=np.vstack([high,[-10,-10],[10,10]]).astype(np.float32)
    landmarks=[dict(id=i,low=a,high=b) for i,(a,b) in enumerate(zip(low,high))]
    index=LandmarkSpatialIndex(landmarks)
    for lo,hi in [([-10,-10],[10,10]),([0,0],[0,0]),([200,200],[210,210])]:
        wanted=[v['id'] for v in landmarks if not (np.any(v['high']<lo) or np.any(v['low']>hi))]
        actual=index.candidates(np.asarray(lo),np.asarray(hi))
        assert [v['id'] for v in actual]==wanted
        assert all(v is landmarks[v['id']] for v in actual)
    assert LandmarkSpatialIndex([]).candidates(np.zeros(2),np.ones(2))==[]
