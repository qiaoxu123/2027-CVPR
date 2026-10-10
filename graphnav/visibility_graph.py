"""Observation-only landmark evidence edges with optional depth validation.

This module is intentionally separated from static ablation inputs.
Only a map landmark that is ACTUALLY in the camera footprint AND has a
reliable ID-matched visual detection gets a positive evidence edge.
No RGB file path, feature, depth or future observation enters static models.
"""

from __future__ import annotations

from dataclasses import dataclass
import cv2
import numpy as np


def _raster_coverage(geometry_world, footprint_world, *, resolution=256):
    """Approximate polygon footprint intersection / landmark area on local raster."""
    contour = np.asarray(geometry_world, np.float32)
    view = np.asarray(footprint_world, np.float32)
    if contour.ndim != 2 or contour.shape[1] != 2 or len(contour) < 3:
        return 0.
    if view.shape != (4, 2):
        raise ValueError("camera footprint must be four observed world corners")
    all_points = np.concatenate((view, contour), axis=0)
    lo, hi = all_points.min(0), all_points.max(0)
    span = np.maximum(hi - lo, 1e-3)
    def pixels(pts):
        return np.rint((pts - lo) / span * (resolution - 1)).astype(np.int32)
    land = cv2.fillPoly(np.zeros((resolution, resolution), np.uint8), [pixels(contour)], 1)
    visible = cv2.fillPoly(np.zeros_like(land), [pixels(view)], 1)
    total = int(land.sum())
    return float((land & visible).sum()) / max(1, total)


@dataclass
class Evidence:
    entity_id: str
    current_coverage: float
    current_match_confidence: float
    historical_match_confidence: float
    observations: int
    depth_observed: bool


class LandmarkVisibilityGraph:
    """Incremental entity-evidence graph with explicit verified matching.

    Identity verification MUST come from a detector/grounder, not simply
    semantic similarity of the entire RGB frame.
    """

    def __init__(self, polygons_by_id, *, min_coverage=.30, min_match_confidence=.70):
        self.polygons = {
            str(name): np.asarray(poly, np.float32) for name, poly in polygons_by_id.items()
        }
        self.min_coverage = min_coverage
        self.min_match_confidence = min_match_confidence
        self.memory = {
            name: Evidence(name, 0., 0., 0., 0, False) for name in self.polygons
        }
        self.edges: dict[str, set[str]] = {name: set() for name in self.polygons}

    def observe(self, footprint_world, matches, *, depth=None, observation_id=None):
        """Update with current *real* observation and identity-matched results.

        matches: {entity_id: independently estimated confidence [0,1]}.
        An out-of-view ID is rejected even if a hallucinating detector scores it 1.
        """
        if depth is not None:
            depth = np.asarray(depth)
            if depth.ndim != 2 or not np.isfinite(depth).any():
                raise ValueError("depth must be a finite 2-D observed array")
        updates = {}
        for entity_id, polygon in self.polygons.items():
            cov = _raster_coverage(polygon, footprint_world)
            confidence = float(matches.get(entity_id, 0.))
            if not 0 <= confidence <= 1:
                raise ValueError("match confidence must be [0,1]")
            old = self.memory[entity_id]
            accepted = cov >= self.min_coverage and confidence >= self.min_match_confidence
            hist = 1. - (1. - old.historical_match_confidence) * (1. - confidence * cov) if accepted else old.historical_match_confidence
            new = Evidence(
                entity_id, cov, confidence if accepted else 0.,
                hist, old.observations + int(accepted),
                old.depth_observed or bool(depth is not None and accepted),
            )
            self.memory[entity_id] = new
            if accepted and observation_id is not None:
                self.edges[entity_id].add(str(observation_id))
            updates[entity_id] = new
        return updates

    def evidence_for(self, entity_id):
        return self.memory[str(entity_id)]
