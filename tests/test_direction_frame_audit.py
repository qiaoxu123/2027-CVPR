import numpy as np
from scripts.audit_direction_frames import interval_status

def test_long_anchor_not_centroid_binary_label():
 p=np.array([[0,0],[100,0],[100,10],[0,10]])
 assert interval_status(p,[10,5],[1,0])[0]=='within_anchor_extent'
 assert interval_status(p,[101,5],[1,0])[0]=='desired_side'
 assert interval_status(p,[-1,5],[1,0])[0]=='opposite_side'

def test_inverse_relation_and_axis_are_explicit():
 p=np.array([[0,0],[10,0],[10,10],[0,10]])
 assert interval_status(p,[5,-1],[0,-1])[0]=='desired_side'
 assert interval_status(p,[5,-1],[0,1])[0]=='opposite_side'

def test_rotation_not_fitted_to_target():
 from scripts.audit_direction_frames import UNITS
 assert UNITS=={'north':(0,1),'south':(0,-1),'east':(1,0),'west':(-1,0)}
