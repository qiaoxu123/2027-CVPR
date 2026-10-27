"""Deterministic world geometry and differentiable polygon relations.

Coordinates use (x east, y north) normalized to the SAME public map bounds.
Polygon vertices are in perimeter order, including when resampled.
"""

from __future__ import annotations

import numpy as np
import torch


def normalize_xy(points, bounds):
    """Map physical coordinates to [0,1] using (xmin,ymin,xmax,ymax)."""
    xy = np.asarray(points, dtype=np.float32)
    lo = np.asarray(bounds[:2], dtype=np.float32)
    hi = np.asarray(bounds[2:], dtype=np.float32)
    if xy.shape[-1] != 2 or np.any(hi <= lo):
        raise ValueError("invalid points or map bounds")
    return (xy - lo) / (hi - lo)


def resample_contour(vertices, count: int = 16):
    """Sample an ordered closed polygon at equal *perimeter* intervals."""
    a = np.asarray(vertices, dtype=np.float32)
    if a.ndim != 2 or a.shape[1] != 2 or len(a) < 3 or count < 3:
        raise ValueError("contour must contain >=3 XY vertices")
    if np.allclose(a[0], a[-1]):
        a = a[:-1]
    nxt = np.roll(a, -1, axis=0)
    lengths = np.linalg.norm(nxt - a, axis=1)
    total = float(lengths.sum())
    if total < 1e-7:
        raise ValueError("degenerate landmark polygon")
    cumulative = np.r_[0.0, np.cumsum(lengths)]
    samples = np.arange(count, dtype=np.float64) * total / count
    indices = np.minimum(np.searchsorted(cumulative, samples, side="right") - 1, len(a) - 1)
    fraction = ((samples - cumulative[indices]) / np.maximum(lengths[indices], 1e-8))
    return (a[indices] + fraction[:, None] * (nxt[indices] - a[indices])).astype(np.float32)


def contour_statistics(contours: torch.Tensor) -> torch.Tensor:
    """Return [B,M,8] geometry: center, bbox extent, area, perimeter, spread."""
    if contours.ndim != 4 or contours.shape[-1] != 2:
        raise ValueError("contours should have shape [B,M,P,2]")
    center = contours.mean(dim=-2)
    extent = contours.amax(dim=-2) - contours.amin(dim=-2)
    nxt = contours.roll(-1, dims=-2)
    cross = contours[..., 0] * nxt[..., 1] - nxt[..., 0] * contours[..., 1]
    area = 0.5 * cross.sum(dim=-1).abs().unsqueeze(-1)
    perimeter = (nxt - contours).norm(dim=-1).sum(dim=-1, keepdim=True)
    spread = (contours - center.unsqueeze(-2)).square().mean(dim=-2).sqrt()
    return torch.cat((center, extent, area, perimeter, spread), dim=-1)


def candidate_polygon_geometry(
    candidates: torch.Tensor,
    contours: torch.Tensor,
    chunk_size: int = 64,
) -> torch.Tensor:
    """[B,Q,M,9] with exact point-to-segment distance and polygon containment.

    Features are candidate-minus-center XY, center distance, bearing cos/sin,
    *signed* minimum contour distance, inside bit, contour width and height.
    No GT position, future observations, or target labels are used.
    """
    if candidates.ndim != 3 or candidates.shape[-1] != 2:
        raise ValueError("candidates must be [B,Q,2]")
    if contours.ndim != 4 or contours.shape[-1] != 2:
        raise ValueError("contours must be [B,M,P,2]")
    if candidates.shape[0] != contours.shape[0]:
        raise ValueError("batch sizes differ")
    if contours.shape[2] < 3:
        raise ValueError("at least 3 contour points required")
    center = contours.mean(dim=2)
    extent = contours.amax(dim=2) - contours.amin(dim=2)
    a = contours[:, None]
    b = contours.roll(-1, dims=2)[:, None]
    ab = b - a
    denominator = (ab * ab).sum(dim=-1).clamp_min(1e-9)
    outputs = []
    for c in candidates.split(chunk_size, dim=1):
        point = c[:, :, None, None, :]
        ap = point - a
        t = ((ap * ab).sum(dim=-1) / denominator).clamp(0.0, 1.0)
        projection = a + t.unsqueeze(-1) * ab
        distance = (point - projection).norm(dim=-1).amin(dim=-1)
        y, x = point[..., 1], point[..., 0]
        ay, by = a[..., 1], b[..., 1]
        ax, bx = a[..., 0], b[..., 0]
        dy = by - ay
        safe_dy = torch.where(dy.abs() < 1e-9, torch.ones_like(dy), dy)
        intersects = ((ay > y) != (by > y)) & (x < (bx - ax) * (y - ay) / safe_dy + ax)
        inside = (intersects.int().sum(dim=-1) % 2).bool()
        signed_distance = torch.where(inside, -distance, distance)
        delta = c[:, :, None, :] - center[:, None, :, :]
        dcenter = delta.norm(dim=-1, keepdim=True).clamp_min(1e-7)
        unit = delta / dcenter
        ext = extent[:, None, :, :].expand(-1, c.shape[1], -1, -1)
        features = torch.cat(
            (
                delta, dcenter, unit,
                signed_distance.unsqueeze(-1), inside.float().unsqueeze(-1), ext,
            ),
            dim=-1,
        )
        outputs.append(features)
    return torch.cat(outputs, dim=1)


def node_edge_geometry(xy: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
    """[B,N,N,8] directed edge geometry, with UAV heading as reference."""
    if xy.ndim != 3 or xy.shape[-1] != 2 or yaw.ndim != 1:
        raise ValueError("expected xy [B,N,2], yaw [B]")
    delta = xy[:, None, :, :] - xy[:, :, None, :]
    dist = delta.norm(dim=-1, keepdim=True).clamp_min(1e-7)
    unit = delta / dist
    bearing = torch.atan2(delta[..., 1], delta[..., 0]) - yaw[:, None, None]
    return torch.cat(
        (
            delta, dist, unit,
            torch.sin(bearing).unsqueeze(-1),
            torch.cos(bearing).unsqueeze(-1),
            dist.square(),
        ),
        dim=-1,
    )


def grid_centers(grid_size: int, *, device=None, dtype=torch.float32) -> torch.Tensor:
    """Return grid [Q,2] in standard Cartesian XY (x,y) order."""
    centers = (torch.arange(grid_size, device=device, dtype=dtype) + 0.5) / grid_size
    yy, xx = torch.meshgrid(centers, centers, indexing="ij")
    return torch.stack((xx, yy), dim=-1).reshape(-1, 2)
