"""Causal observed-only RGB mosaic for aerial navigation.

Never reads the scene's underlying orthomosaic. Pixels enter this structure
ONLY when an observation arrives, with its measured/predicted camera footprint.
"""

from __future__ import annotations

from dataclasses import dataclass
import cv2
import numpy as np


@dataclass(frozen=True)
class VisibleCrop:
    image: np.ndarray             # [size,size,3] uint8
    valid: np.ndarray             # [size,size] bool
    coverage: float               # valid / total sampled pixels
    center_coverage: float        # target-centered local valid ratio


class ObservedRGBMemory:
    def __init__(self, world_bounds, size=512):
        """world_bounds = (xmin,ymin,xmax,ymax) in meters; x east/y north."""
        self.bounds = np.asarray(world_bounds, dtype=np.float32)
        if self.bounds.shape != (4,) or np.any(self.bounds[2:] <= self.bounds[:2]):
            raise ValueError("invalid world bounds")
        if size < 16:
            raise ValueError("memory image should be at least 16x16")
        self.size = int(size)
        self.rgb = np.zeros((size, size, 3), dtype=np.uint8)
        self.seen = np.zeros((size, size), dtype=np.uint8)
        self.current = np.zeros((size, size), dtype=np.uint8)
        self.version = 0

    def world_to_image(self, xy):
        xy = np.asarray(xy, dtype=np.float32)
        x0, y0, x1, y1 = self.bounds
        u = (xy[..., 0] - x0) / (x1 - x0) * (self.size - 1)
        v = (y1 - xy[..., 1]) / (y1 - y0) * (self.size - 1)
        return np.stack((u, v), axis=-1).astype(np.float32)

    @staticmethod
    def view_corners(xy, yaw, altitude_above_ground):
        """CityNav-style square nadir camera, source image TL,TR,BR,BL."""
        if altitude_above_ground <= 0:
            raise ValueError("relative altitude must be positive")
        xy = np.asarray(xy, dtype=np.float32)
        front = np.asarray((np.cos(yaw), np.sin(yaw)), dtype=np.float32)
        left = np.asarray((-np.sin(yaw), np.cos(yaw)), dtype=np.float32)
        h = float(altitude_above_ground)
        return np.stack((
            xy + h * (front + left),
            xy + h * (front - left),
            xy - h * (front + left),
            xy - h * (front - left),
        )).astype(np.float32)

    def observe(self, rgb, footprint_world):
        """Project one actual top-down RGB frame into world space.

        footprint_world must list *pixel* corners TL,TR,BR,BL in world XY.
        There is deliberately no method to prefill unobserved pixels.
        """
        image = np.asarray(rgb)
        corners = np.asarray(footprint_world, dtype=np.float32)
        if image.ndim != 3 or image.shape[-1] != 3 or image.dtype != np.uint8:
            raise ValueError("RGB must be HxWx3 uint8")
        if corners.shape != (4, 2):
            raise ValueError("footprint must contain four world XY corners")
        h, w = image.shape[:2]
        source = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
        dest = self.world_to_image(corners)
        transform = cv2.getPerspectiveTransform(source, dest)
        size = (self.size, self.size)
        warped = cv2.warpPerspective(
            image, transform, size, flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0),
        )
        footprint = cv2.warpPerspective(
            np.ones((h, w), dtype=np.uint8), transform, size,
            flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0,
        )
        fresh = footprint != 0
        self.rgb[fresh] = warped[fresh]
        self.current = footprint
        self.seen[fresh] = 1
        if fresh.any():
            self.version += 1
        return int(fresh.sum())

    def crop(self, center_world, extent_m=100.0, size=224) -> VisibleCrop:
        """Sample an axis-aligned target-centered patch from observed pixels only."""
        if extent_m <= 0 or size < 8:
            raise ValueError("invalid crop extent or size")
        center = self.world_to_image(center_world)
        x0, y0, x1, y1 = self.bounds
        step_x = extent_m * (self.size - 1) / ((x1 - x0) * (size - 1))
        step_y = extent_m * (self.size - 1) / ((y1 - y0) * (size - 1))
        grid = np.arange(size, dtype=np.float32) - (size - 1) / 2.0
        xs = center[0] + grid * step_x
        ys = center[1] + grid * step_y
        map_x, map_y = np.meshgrid(xs, ys, indexing="xy")
        kwargs = dict(interpolation=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT)
        valid = cv2.remap(
            self.seen, map_x, map_y, borderValue=0, **kwargs
        ).astype(bool)
        patch = cv2.remap(
            self.rgb, map_x, map_y, interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0),
        )
        patch[~valid] = 0
        r0, r1 = size * 3 // 8, size * 5 // 8
        center_coverage = float(valid[r0:r1, r0:r1].mean())
        return VisibleCrop(patch, valid, float(valid.mean()), center_coverage)

    def masks(self, grid_size: int):
        """Return [2,G,G] = current and accumulated *actual* observation masks."""
        return np.stack(
            [
                cv2.resize(self.current, (grid_size, grid_size), interpolation=cv2.INTER_AREA),
                cv2.resize(self.seen, (grid_size, grid_size), interpolation=cv2.INTER_AREA),
            ],
            axis=0,
        ).astype(np.float32)

    @property
    def explored_fraction(self):
        return float(self.seen.mean())
