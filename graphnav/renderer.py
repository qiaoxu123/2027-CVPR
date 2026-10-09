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
        return {"path": str(save_path.resolve()), "footprint_world": footprint.tolist()}
