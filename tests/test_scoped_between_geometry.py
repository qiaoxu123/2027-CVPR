import numpy as np
from graphnav.scoped_between_geometry import between_distance


def test_public_hull_does_not_use_private_positions_and_keeps_alternatives():
    objs={str(i):dict(contour=[[i,0],[i+1,0],[i+1,1],[i,1]],position=[999,999]) for i in (0,4,10)}
    endpoints=[dict(public_ids=['0','10']),dict(public_ids=['4'])]
    xy=[[2,.5],[8,.5],[5,3]]
    d,known=between_distance(xy,endpoints,objs)
    assert known and np.allclose(d[:2],0) and d[2]>0
    objs['0']['position']=[-12345,12]
    assert np.array_equal(d,between_distance(xy,endpoints,objs)[0])


def test_invalid_or_same_object_pairs_abstain_with_neutral_distance():
    objs={'0':dict(contour=[[0,0],[1,0],[1,1]])}
    d,known=between_distance([[20,30]],[dict(public_ids=['0']),dict(public_ids=['0'])],objs)
    assert not known and d.tolist()==[0]
