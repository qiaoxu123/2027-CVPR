"""PRIVILEGED diagnostic: complete landmark RGB from the CityNav source orthophoto.

This does NOT represent an observed UAV camera frame. Source RGB is read only
by the offline experimental driver, never by the normal closed-loop policy.
A public, uniquely selected Polygon determines the crop; no GT target,
teacher pose, predicted goal, future trajectory or task outcome enters it.

A 384x384, 24x24-patch rendering is intentional: it uses exactly the current
PGVB/SigLIP2 interface. Huge footprints may make small targets sub-patch.
"""
from __future__ import annotations

import cv2
import numpy as np
import shapely.geometry

from .landmark_region_supervision import named_union
from .memory import ObservedRGBMemory


def whole_landmark_view(contours, *, margin_fraction=0.25, min_extent_m=12.0):
    """Derive one square raster footprint enclosing ALL public Polygon parts.

    The pseudo-camera is purely a map rectification frame, not a flight pose.
    """
    if not 0 <= margin_fraction <= 1 or min_extent_m <= 0:
        raise ValueError("Invalid landmark view extent controls")
    region = named_union(contours)
    if region.is_empty or not region.is_valid or region.area <= 0:
        raise ValueError("Nonempty valid public Polygon required")
    west, south, east, north = map(float, region.bounds)
    if not np.isfinite([west, south, east, north]).all():
        raise ValueError("Nonfinite landmark bounds")
    center = np.array([(west + east) / 2, (south + north) / 2], np.float32)
    extent = max(east-west, north-south, float(min_extent_m))
    # Cropping half-side > half the largest bbox dimension ensures margin.
    half_side = (0.5 + margin_fraction) * extent
    footprint = ObservedRGBMemory.view_corners(center, 0.0, half_side)
    coverage = region.intersection(shapely.geometry.Polygon(footprint)).area / region.area
    if coverage < 0.999:
        raise AssertionError("Complete landmark crop must cover whole public Polygon")
    return dict(
        footprint_world=footprint, center_world=center,
        side_m=2.0 * half_side,
        polygon_fraction=float(coverage),
        public_polygon_area_m2=float(region.area),
        nominal_patch_m=float(2.0 * half_side / 24.0),
    )


def privileged_whole_landmark_rgb(renderer, map_name, contours, *,
                                  margin_fraction=0.25):
    """Read the source orthophoto in a separate OFFLINE analysis boundary.

    Do not pass this output into a benchmark-compatible causal policy and
    do not label its area footprint as an observed UAV image.
    """
    plan = whole_landmark_view(contours, margin_fraction=margin_fraction)
    renderer._open(map_name)  # explicitly privileged; normal renderer API not used
    raster = renderer._raster
    image = renderer._rgb
    if raster is None or image is None or raster.count < 1:
        raise ValueError("Missing complete map raster")
    corners = plan["footprint_world"]
    pixel_corners = np.asarray([
        [raster.index(float(x), float(y))[1],
         raster.index(float(x), float(y))[0]]
        for x, y in corners
    ], np.float32)
    n = 384
    dest = np.float32([[0, 0], [n-1, 0], [n-1, n-1], [0, n-1]])
    matrix = cv2.getPerspectiveTransform(pixel_corners, dest)
    bgr = cv2.warpPerspective(image, matrix, (n, n), flags=cv2.INTER_LINEAR)

    # This is source validity, NOT range/depth visibility. Raster valid bits
    # prevent invalid height pixels from becoming falsely usable patches.
    height = raster.read(1)
    source_valid = (
        np.isfinite(height) & (height >= 0) & (raster.read_masks(1) > 0)
    ).astype(np.float32)
    support = cv2.warpPerspective(
        source_valid, matrix, (n, n), flags=cv2.INTER_LINEAR
    ) >= 0.999
    support &= bgr.max(-1) > 0
    observed = dict(
        bgr=bgr, depth_valid=support,
        # The encoder does not consume depth_m, but the mapping protocol
        # includes this field. Fill NaNs rather than fabricate metric depth.
        depth_m=np.full((n, n), np.nan, np.float32),
        footprint_world=corners.copy(),
        provenance="PRIVILEGED_SOURCE_ORTHOPHOTO_NOT_UAV_OBSERVATION",
        pseudo_view=True,
    )
    details = dict(
        polygon_fraction=plan["polygon_fraction"],
        polygon_area_m2=plan["public_polygon_area_m2"],
        nominal_patch_m=plan["nominal_patch_m"],
        side_m=plan["side_m"],
        valid_source_fraction=float(support.mean()),
        map_rgb_pixels=int(support.sum()),
        source="full_map_rgb_privileged",
    )
    return observed, details
