"""Opt-in camera sampling adapter; no changes to navigation pose or legacy renderer.

The wrapped environment still owns the raster. Policies receive actual crops only.
Camera yaw=0 means local +X/east, not north-up in CityNav's convention.
"""
from __future__ import annotations

import numpy as np


class FixedYawObservation:
    def __init__(self, renderer, yaw_rad=0.):
        if not np.isfinite(yaw_rad):
            raise ValueError('finite fixed camera yaw required')
        self.renderer = renderer
        self.yaw_rad = float(yaw_rad)

    def current_rgbd(self, map_name, pose, ground_level):
        agent_pose = np.asarray(pose, dtype=np.float64)
        if agent_pose.shape != (4,) or not np.isfinite(agent_pose).all():
            raise ValueError('finite agent xyzyaw required')
        camera_pose = agent_pose.copy()
        camera_pose[3] = self.yaw_rad
        observation = dict(self.renderer.current_rgbd(map_name, camera_pose, ground_level))
        observation.update(agent_pose=agent_pose.copy(), agent_yaw_rad=float(agent_pose[3]),
                           camera_yaw_rad=self.yaw_rad, camera_sampling='fixed-yaw-v1')
        return observation
