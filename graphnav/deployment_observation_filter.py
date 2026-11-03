"""TRAIN sampling mask from public pose and frozen action envelope only."""
import math


def observation_in_envelope(frame,ground_level,min_height=10.,max_height=100.):
    height=float(frame['agent_pose'][2])-float(ground_level)
    if not math.isfinite(height) or not 0<min_height<=max_height:raise ValueError('finite height and valid action bounds required')
    if height<=0:return False
    # The benchmark supplies the initial pose; actions need not be able to recreate it.
    if frame['step']==0:return True
    return min_height-1e-4<=height<=max_height+1e-4


def aligned_observation_pose(frame,ground_level,min_height=10.,max_height=100.):
    observation_in_envelope(frame,ground_level,min_height,max_height)
    pose=list(frame['agent_pose'])
    if len(pose)!=4 or not all(math.isfinite(float(x)) for x in pose):raise ValueError('finite xyzyaw pose required')
    if frame['step']!=0:pose[2]=float(ground_level)+min(max(float(pose[2])-float(ground_level),min_height),max_height)
    return pose
