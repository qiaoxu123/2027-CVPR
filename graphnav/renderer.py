"""Offline CityNav observed-view rendering. NOT accessible to model inference.

The offline dataset tool reads the source orthophoto to synthesize *only* the
camera view from each visited UAV pose, using the same four-corner warp used
in the CityNav crop client. GraphNav itself receives only these saved views.
"""

from __future__ import annotations

from pathlib import Path
import cv2
import numpy as np

from .memory import ObservedRGBMemory


class CityNavObservedViewRenderer:
    def __init__(self, rgbd_directory, output_directory, image_size=224):
        self.rgbd = Path(rgbd_directory)
        self.output = Path(output_directory)
        self.image_size = int(image_size)
        self._scene_name = None
        self._raster = None
        self._rgb = None
        self._height = None
        self._height_valid = None

    def __del__(self):
        if self._raster is not None:
            self._raster.close()

    def _open(self, scene_name):
        if scene_name == self._scene_name:
            return
        import rasterio

        raster_path = self.rgbd / f"{scene_name}.tif"
        rgb_path = self.rgbd / f"{scene_name}.png"
        if not raster_path.is_file() or not rgb_path.is_file():
            raise FileNotFoundError(f"CityNav raster/RGB missing for {scene_name}")
        new_raster = rasterio.open(raster_path)
        rgb = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
        if rgb is None:
            new_raster.close()
            raise FileNotFoundError(rgb_path)
        if self._raster is not None:
            self._raster.close()
        self._scene_name = scene_name
        self._raster = new_raster
        self._rgb = rgb  # BGR, because cv2.imwrite accepts BGR
        self._height = None
        self._height_valid = None

    def current_rgbd(self, map_name, pose, ground_level):
        """Native current orthographic RGB and vertical-depth proxy, without disk I/O.

        The benchmark supplies an orthophoto and surface-height raster. Depth is
        camera z minus that surface, NOT perspective ray depth or an occlusion
        renderer. Unknown/above-camera source pixels are invalid, not free space.
        """
        self._open(map_name)
        pose = np.asarray(pose, dtype=np.float64)
        if pose.shape != (4,) or not np.isfinite(pose).all():
            raise ValueError('current pose must be finite xyzyaw')
        footprint = ObservedRGBMemory.view_corners(pose[:2],pose[3],pose[2]-ground_level)
        source = np.asarray([[self._raster.index(float(x),float(y))[1],
                              self._raster.index(float(x),float(y))[0]] for x,y in footprint],np.float32)
        n = self.image_size
        destination = np.float32([[0,0],[n-1,0],[n-1,n-1],[0,n-1]])
        matrix = cv2.getPerspectiveTransform(source,destination)
        rgb = cv2.warpPerspective(self._rgb,matrix,(n,n),flags=cv2.INTER_LINEAR)
        if self._height is None:
            raw=self._raster.read(1).astype(np.float32)
            source_valid=np.isfinite(raw)&(raw>=0)
            self._height_valid=source_valid.astype(np.float32)
            self._height=np.where(source_valid,raw,0)
        support=cv2.warpPerspective(self._height_valid,matrix,(n,n),
                                    flags=cv2.INTER_LINEAR)
        valid=support >= .999
        height=cv2.warpPerspective(self._height,matrix,(n,n),
                                   flags=cv2.INTER_LINEAR)
        valid &= (rgb.max(-1)>0)&(height<pose[2])
        depth=np.where(valid,pose[2]-height,0).astype(np.float32)
        return {'bgr':rgb,'depth_m':depth,'depth_valid':valid,
                'surface_height_m':height,'footprint_world':footprint,'pose':pose.copy()}

    def render(self, map_name, episode_key, step, pose, ground_level):
        """Create exactly one current 224x224 view and its world footprint."""
        self._open(map_name)
        if len(pose) == 5:
            yaw = float(pose[3])
        elif len(pose) == 6:
            yaw = float(np.arctan2(pose[4], pose[3]))
        else:
            raise ValueError("invalid CityNav pose")
        altitude = float(pose[2]) - float(ground_level)
        footprint = ObservedRGBMemory.view_corners(pose[:2], yaw, altitude)
        px = np.asarray(
            [[self._raster.index(float(x), float(y))[1],
              self._raster.index(float(x), float(y))[0]]
             for x, y in footprint],
            dtype=np.float32,
        )
        n = self.image_size
        corners = np.float32([[0, 0], [n - 1, 0], [n - 1, n - 1], [0, n - 1]])
        matrix = cv2.getPerspectiveTransform(px, corners)
        image = cv2.warpPerspective(
            self._rgb, matrix, (n, n), flags=cv2.INTER_LINEAR
        )
        save_path = self.output / map_name / episode_key / f"{step:03d}.png"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(save_path), image):
            raise IOError(f"failed to save observation {save_path}")
        return {"path": str(save_path.resolve()), "footprint_world": footprint.tolist(), "step": int(step)}
