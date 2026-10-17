"""Shared evaluation protocol. Labels are separated from the policy interface."""
from dataclasses import dataclass
import math
import numpy as np


@dataclass(frozen=True)
class PublicEpisode:
    map_name: str
    instruction: str
    start_pose: np.ndarray
    references: tuple
    reference_names: tuple = ()


@dataclass(frozen=True)
class ClosedLoopProtocol:
    horizon: int = 20
    horizontal_step_m: float = 50.0
    vertical_step_m: float = 10.0
    min_altitude_m: float = 10.0
    max_altitude_m: float = 100.0
    stagnation_steps: int = 5
    stagnation_tolerance_m: float = 1e-4
    success_radius_m: float = 20.0
    dimensions: int = 2


def score_trace(poses, target, reference, termination, *, radius=20.0,
                decision_seconds=()):
    """Released benchmark XY SR/SPL; target/reference accessed after rollout only."""
    path = np.asarray(poses, dtype=float)[:, :2]
    target = np.asarray(target, dtype=float)[:2]
    reference = np.asarray(reference, dtype=float)[:, :2]
    if not len(path) or not len(reference) or radius <= 0:
        raise ValueError("nonempty paths and positive success radius required")
    if not all(np.isfinite(x).all() for x in (path, target, reference)):
        raise ValueError("nonfinite evaluation coordinates")
    distances = np.linalg.norm(path-target, axis=1)
    length = np.linalg.norm(np.diff(path, axis=0), axis=1).sum()
    optimal = distances[0]
    success = float(distances[-1] <= radius)
    previous = np.full(len(reference)+1, np.inf); previous[0] = 0
    for point in path:
        current = np.full(len(reference)+1, np.inf)
        costs = np.linalg.norm(reference-point, axis=1)
        for j, cost in enumerate(costs, 1):
            current[j] = cost+min(current[j-1], previous[j], previous[j-1])
        previous = current
    ndtw = math.exp(-previous[-1]/(radius*len(reference)))
    stop = termination == "policy_stop"
    return dict(SR=success, OSR=float(distances.min() <= radius),
                SPL=float(success*optimal/max(length, optimal, 1e-9)),
                NE=float(distances[-1]), nDTW=ndtw, sDTW=success*ndtw,
                steps=len(path)-1, path_length=float(length),
                policy_stop=float(stop), false_stop=float(stop and not success),
                inference_ms=float(np.mean(decision_seconds)*1000) if decision_seconds else 0.0)


def aggregate(rows):
    if not rows:
        raise ValueError("empty evaluation")
    result = {key: float(np.mean([row[key] for row in rows])) for key in rows[0]}
    result['episodes'] = len(rows)
    stopped = sum(row['policy_stop'] for row in rows)
    result['False_Stop_Rate'] = sum(row['false_stop'] for row in rows)/stopped if stopped else None
    result['false_stop_episode_fraction'] = result.pop('false_stop')
    result['metric_units'] = "SR/OSR/SPL/nDTW/sDTW fractions; NE meters; inference_ms milliseconds"
    return result


class PublicCatalog:
    """Pass only resolved language/map references to the observation builder."""
    def __init__(self, catalog):
        self.catalog = catalog

    def all_landmarks(self, map_name):
        return self.catalog.all_landmarks(map_name)

    def referenced_landmarks(self, episode):
        return episode.references


def rollout(episode, builder, controller, policy, ground, protocol, *, synchronize=lambda: None, profiler=None):
    import time
    from contextlib import nullcontext
    poses = [episode.start_pose.copy()]
    timings, diagnostics = [], []
    stagnant = 0
    termination = 'horizon'
    transform = builder.transform(episode.map_name)
    bounds = (transform.x_min, transform.y_min, transform.x_max, transform.y_max)
    for step in range(protocol.horizon):
        with profiler.sample(episode.map_name, step) if profiler else nullcontext():
            state = builder.build_from_poses(episode, np.stack(poses))
            synchronize(); started = time.perf_counter()
            prediction = policy(state, episode.instruction)
            synchronize(); timings.append(time.perf_counter()-started)
            diagnostics.append(prediction.diagnostics)
            if getattr(prediction, 'should_stop', False):
                termination = 'policy_stop'; break
            next_pose = controller.step(poses[-1], prediction.xyz,
                                        ground_level=ground, xy_bounds=bounds)
            displacement = np.linalg.norm(next_pose[:3]-poses[-1][:3])
            poses.append(next_pose)
            stagnant = stagnant+1 if displacement <= protocol.stagnation_tolerance_m else 0
            if stagnant >= protocol.stagnation_steps:
                termination = 'stagnation'; break
    return np.stack(poses), termination, timings, diagnostics
