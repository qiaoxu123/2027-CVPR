"""Deterministic spatial diversity for a bounded candidate search."""

import numpy as np


def spatial_candidate_slots(candidate_world_xy, *, max_candidates, min_distance_m):
    """Return score-ordered candidate slots separated in world XY.

    ``candidate_world_xy`` must already be ordered by the model's public
    Geometry score. No labels or target coordinates are consulted.
    """
    points = np.asarray(candidate_world_xy, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("candidate_world_xy must have shape [K,2]")
    if not np.isfinite(points).all():
        raise ValueError("candidate_world_xy must be finite")
    if max_candidates < 0:
        raise ValueError("max_candidates must be nonnegative")
    if min_distance_m < 0:
        raise ValueError("min_distance_m must be nonnegative")
    if max_candidates == 0:
        return []
    kept = []
    for slot, point in enumerate(points):
        if all(np.linalg.norm(point - points[prior]) >= min_distance_m for prior in kept):
            kept.append(slot)
            if len(kept) >= max_candidates:
                break
    return kept
