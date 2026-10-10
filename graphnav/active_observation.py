"""Optional active-perception policy for the *existing* CityNav nadir benchmark.

A causal, map-only proposal planner. It changes only bounded XY/Z waypoints,
never labels, camera imagery, predicted goals, evaluator success or yaw.
Its score is a hypothesis for ablations, NOT a learned information-gain model
or a claim of improved semantic recognition.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import re

import numpy as np


@dataclass(frozen=True)
class ActiveObservationConfig:
    max_actions: int = 2
    horizon: int = 20
    min_remaining_actions: int = 4
    horizontal_step_m: float = 50.0
    vertical_step_m: float = 10.0
    min_height_m: float = 10.0
    max_height_m: float = 100.0
    approach_fraction: float = 0.8
    lateral_m: float = 12.0
    min_pixel_diameter: float = 16.0
    enable_approach: bool = True
    enable_descend: bool = True
    enable_lateral: bool = True


@dataclass(frozen=True)
class ActiveObservationDecision:
    action: str
    xyz: np.ndarray
    candidate_id: int
    candidate_world_xy: tuple[float, float]
    estimated_object_pixels: float
    observed_confidence: float
    reason: str


def appearance_size_proxy_m(instruction: str) -> float:
    """Coarse reference size, *not* the unknown real object diameter.

    A small visible reference, e.g. a car next to the target building, can
    dictate required observation scale. The proxy is deliberately logged.
    """
    if not isinstance(instruction, str):
        raise TypeError("instruction must be str")
    if re.search(r"\b(car|cars|vehicle|vehicles|skylight|chimney)\b", instruction, re.I):
        return 4.0
    if re.search(r"\b(house|building|roof|warehouse)\b", instruction, re.I):
        return 15.0
    return 8.0


def estimated_object_pixels(proxy_size_m: float, altitude_m: float, image_size: int = 384):
    """Ideal nadir sampling only; source raster resolution may be much worse."""
    if proxy_size_m <= 0 or altitude_m <= 0 or image_size <= 0:
        raise ValueError("positive proxy, altitude and image size are required")
    return float(proxy_size_m * image_size / (2.0 * altitude_m))


def _bounded_step(current: np.ndarray, goal: np.ndarray, maximum: float) -> np.ndarray:
    move = goal - current
    norm = float(np.linalg.norm(move))
    return current + move * min(1.0, maximum / max(norm, 1e-10))


def plan_active_observation(
    *,
    pose_xyzyaw,
    map_bounds,
    ground_level: float,
    candidate_xy,
    candidate_logits,
    observed_confidence,
    instruction: str,
    step: int,
    used_actions: int = 0,
    config: ActiveObservationConfig = ActiveObservationConfig(),
):
    """Return an optional bounded exploratory move, or None.

    All candidates are normalized public-map XY. Confidence represents actual
    evidence *already observed*, not GT, target occupancy or true semantic
    visibility. Need and action ordering are fixed heuristics, pre-declared for
    fair ablation and subject to a conservative action budget.
    """
    pose = np.asarray(pose_xyzyaw, dtype=np.float64)
    bounds = np.asarray(map_bounds, dtype=np.float64)
    cells = np.asarray(candidate_xy, dtype=np.float64)
    scores = np.asarray(candidate_logits, dtype=np.float64).reshape(-1)
    seen = np.asarray(observed_confidence, dtype=np.float64).reshape(-1)
    if pose.shape != (4,) or bounds.shape != (4,) or cells.shape != (len(scores), 2):
        raise ValueError("pose, bounds or candidate grid shape mismatch")
    if seen.shape != (len(scores),) or not len(scores):
        raise ValueError("observation confidence must match nonempty candidate grid")
    if any(not np.isfinite(t).all() for t in (pose, bounds, cells, scores, seen)):
        raise ValueError("nonfinite active observation inputs")
    if not np.isfinite(ground_level) or np.any(bounds[2:] <= bounds[:2]):
        raise ValueError("invalid public ground/bounds")
    if np.any((cells < 0) | (cells > 1)) or np.any((seen < 0) | (seen > 1)):
        raise ValueError("candidate XY / observed confidence must lie in [0,1]")
    if config.max_actions < 0 or config.horizon < 1 or config.min_remaining_actions < 1:
        raise ValueError("invalid action budget")
    if config.horizontal_step_m <= 0 or config.vertical_step_m <= 0 or config.lateral_m < 0:
        raise ValueError("invalid controller bounds")
    if not 0 < config.approach_fraction <= 1 or config.min_pixel_diameter <= 0:
        raise ValueError("invalid sensing parameters")
    if not 0 < config.min_height_m < config.max_height_m:
        raise ValueError("invalid altitude limits")
    if used_actions >= config.max_actions or config.horizon - step < config.min_remaining_actions:
        return None
    height = float(pose[2] - ground_level)
    # Initial source starts can be outside legal altitude. Let the original
    # bounded controller recover them; never impose a new active move there.
    if not config.min_height_m <= height <= config.max_height_m:
        return None
    if np.any(pose[:2] < bounds[:2]) or np.any(pose[:2] > bounds[2:]):
        # Clipping an invalid start to the map can exceed the single-step cap.
        # Preserve it and let the original bounded controller recover instead.
        return None

    # Compare only the strongest few modes and prefer those without evidence.
    top = np.argsort(-scores, kind="stable")[: min(8, len(scores))]
    shifted = scores[top] - np.max(scores[top])
    weights = np.exp(shifted)
    weights /= weights.sum()
    # Avoid extra active moves when all the top hypotheses have already been
    # observed, or the high-belief best candidate is strongly resolved.
    if float(np.max(1.0 - seen[top])) < 0.15:
        return None
    if len(top) > 1 and weights[0] > 0.9 and seen[top[0]] > 0.6:
        return None
    reward = weights * (1.0 - seen[top])
    chosen = int(top[int(reward.argmax())])
    goal = bounds[:2] + cells[chosen] * (bounds[2:] - bounds[:2])
    distance = float(np.linalg.norm(goal - pose[:2]))
    proxy = appearance_size_proxy_m(instruction)
    pixels = estimated_object_pixels(proxy, height)

    def result(action, target_xy, target_z, reason):
        # All actions are at most one *public* macro step. The benchmark
        # controller remains the final authority for step/bounds/ground.
        xy = _bounded_step(pose[:2], np.asarray(target_xy, dtype=np.float64),
                           config.horizontal_step_m)
        xy = np.clip(xy, bounds[:2], bounds[2:])
        z = float(np.clip(target_z, max(ground_level + config.min_height_m,
                                       pose[2] - config.vertical_step_m),
                          min(ground_level + config.max_height_m,
                              pose[2] + config.vertical_step_m)))
        if np.linalg.norm(xy - pose[:2]) < 1e-6 and abs(z - pose[2]) < 1e-6:
            return None
        return ActiveObservationDecision(
            action=action, xyz=np.array([xy[0], xy[1], z], dtype=np.float64),
            candidate_id=chosen,
            candidate_world_xy=(float(goal[0]), float(goal[1])),
            estimated_object_pixels=pixels,
            observed_confidence=float(seen[chosen]),
            reason=reason,
        )

    # A candidate far outside the inscribed nadir footprint should be
    # approached before spending an action on height/viewpoint adjustment.
    if config.enable_approach and distance > height * config.approach_fraction:
        return result("approach", goal, pose[2],
                      "bring an unobserved high-belief region into current nadir FOV")
    # Height changes alter physical sampling; digital ROI upsampling cannot.
    if (config.enable_descend and pixels < config.min_pixel_diameter
            and height > config.min_height_m + config.vertical_step_m / 2):
        answer = result("descend", pose[:2], pose[2] - config.vertical_step_m,
                        "increase ideal pixels per small visual reference")
        if answer is not None:
            return answer
    # Lateral relocation changes nadir footprint coverage, NOT 3-D parallax.
    if config.enable_lateral and distance < height * 1.5 and config.lateral_m > 0:
        direction = goal - pose[:2]
        if np.linalg.norm(direction) < 1e-6:
            direction = np.array([math.cos(pose[3]), math.sin(pose[3])])
        unit = direction / np.linalg.norm(direction)
        sideways = np.array([-unit[1], unit[0]])
        sign = -1.0 if used_actions % 2 else 1.0
        return result("lateral", pose[:2] + sign * sideways * config.lateral_m,
                      pose[2], "seek new source-pixel footprint without claiming parallax")
    return None
