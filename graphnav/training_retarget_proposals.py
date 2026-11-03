"""Private TRAIN teacher proposals only; no normal-inference use permitted."""
import numpy as np
from scipy.spatial import cKDTree


def registered_core_proposals(candidate_world, valid, object_positions, object_ids,
                              radius_m=2., neighbor_margin_m=.75):
    world = np.asarray(candidate_world, float); positions = np.asarray(object_positions, float)
    valid = np.asarray(valid); ids = list(object_ids)
    if world.ndim != 2 or world.shape[-1] != 2 or valid.shape != (len(world),) or valid.dtype != bool:
        raise ValueError('aligned observed hypotheses and validity required')
    if positions.shape != (len(ids), 2) or len(set(ids)) != len(ids):
        raise ValueError('unique registered TRAIN instances required')
    if not np.isfinite(world).all() or not np.isfinite(positions).all() or radius_m <= 0 or neighbor_margin_m < 0:
        raise ValueError('finite coordinates and valid uncertainty settings required')
    if not ids: return {}
    distance, nearest = cKDTree(positions).query(world, k=2)
    selected = valid & (distance[:, 0] <= radius_m) & (distance[:, 1]-distance[:, 0] >= neighbor_margin_m)
    output = {}
    for candidate in np.flatnonzero(selected):
        output.setdefault(ids[int(nearest[candidate, 0])], []).append(int(candidate))
    return output
