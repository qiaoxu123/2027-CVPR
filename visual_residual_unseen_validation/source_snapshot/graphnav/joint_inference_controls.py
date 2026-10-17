"""Pure, GT-free inference gates for joint GraphNav closed-loop ablations.

Policy must use only selected *predicted* candidate, public UAV pose,
public altitude and the trained Stop logit. No target/teacher labels here.
"""
from __future__ import annotations

import numpy as np


def joint_stop_decision(
    *,
    pose_world,
    candidate_world_xy,
    stop_probability: float,
    ground_level: float,
    threshold: float = 0.5,
    guard_radius_m: float = 20.0,
    min_altitude_m: float = 10.0,
    max_altitude_m: float = 100.0,
):
    """Return (should_stop, diagnostics), without modifying the learned head.

    guard_radius_m=0 explicitly disables only the predicted-goal distance
    guard; ground-relative legal altitude remains mandatory. The 20 m guard
    is a proposed control ablation, not a certified optimal threshold.
    """
    pose = np.asarray(pose_world, dtype=np.float64)
    goal = np.asarray(candidate_world_xy, dtype=np.float64)
    if pose.shape != (4,) or goal.shape != (2,):
        raise ValueError("public pose [x,y,z,yaw] and selected predicted goal [x,y] required")
    if not (np.isfinite(pose).all() and np.isfinite(goal).all()):
        raise ValueError("nonfinite public pose or predicted candidate")
    scalar = np.array(
        [stop_probability, ground_level, threshold, guard_radius_m,
         min_altitude_m, max_altitude_m], dtype=np.float64,
    )
    if not np.isfinite(scalar).all() or not 0 <= stop_probability <= 1:
        raise ValueError("invalid finite Stop parameters")
    if not 0 <= threshold <= 1 or guard_radius_m < 0:
        raise ValueError("invalid Stop threshold or predicted-goal guard")
    if min_altitude_m >= max_altitude_m:
        raise ValueError("invalid public altitude limits")
    predicted_distance = float(np.linalg.norm(pose[:2] - goal))
    altitude = float(pose[2] - ground_level)
    probability_ok = bool(stop_probability >= threshold)
    altitude_ok = bool(min_altitude_m - 1e-5 <= altitude <= max_altitude_m + 1e-5)
    goal_ok = bool(guard_radius_m == 0 or predicted_distance <= guard_radius_m)
    should_stop = probability_ok and altitude_ok and goal_ok
    return should_stop, dict(
        stop_probability=float(stop_probability),
        stop_threshold=float(threshold),
        stop_predicted_goal_distance_m=predicted_distance,
        stop_goal_guard_radius_m=float(guard_radius_m),
        stop_probability_pass=probability_ok,
        stop_altitude_guard_pass=altitude_ok,
        stop_predicted_goal_guard_pass=goal_ok,
        stop_reject_reason=(
            "none" if should_stop
            else "probability" if not probability_ok
            else "altitude" if not altitude_ok
            else "predicted_goal_distance"
        ),
    )
