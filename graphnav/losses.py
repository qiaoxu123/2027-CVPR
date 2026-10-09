"""Training objectives; target positions and teacher paths are supervision only."""

from __future__ import annotations
import torch
import torch.nn.functional as F

from .config import GraphNavConfig
from .geometry import grid_centers
from .model import GraphNavOutput


def goal_distribution(target_xy: torch.Tensor, grid_size: int, sigma_cells=0.9):
    grid = grid_centers(grid_size, device=target_xy.device, dtype=target_xy.dtype)
    sigma = sigma_cells / grid_size
    dist2 = (grid.unsqueeze(0) - target_xy[:, None]).square().sum(dim=-1)
    probs = torch.softmax(-0.5 * dist2 / (sigma * sigma), dim=-1)
    return probs


def navigation_losses(
    pred: GraphNavOutput,
    target_xy: torch.Tensor,
    uav_pose: torch.Tensor,
    config: GraphNavConfig,
    teacher_future: torch.Tensor | None = None,
    physical_map_sizes: torch.Tensor | None = None,
    stop_radius_m: float = 20.0,
    dense_weight=1.0,
    trajectory_weight=0.5,
    ranking_weight=0.1,
    stop_weight=0.1,
):
    """Soft goal CE + min-of-modes path imitation + candidate ranking + stop BCE.

    teacher_future is genuine teacher path, NOT a straight line generated from
    target labels; omit when unavailable. Inference never receives target_xy.
    """
    target = goal_distribution(target_xy, config.grid_size)
    goal_loss = -(target * F.log_softmax(pred.belief_logits, dim=-1)).sum(dim=-1).mean()
    losses = {"goal": goal_loss}
    total = dense_weight * goal_loss

    nearest_idx = (pred.candidate_xy - target_xy[:, None]).square().sum(-1).argmin(-1)
    batch_idx = torch.arange(target_xy.shape[0], device=target_xy.device)
    # Candidate identity objective is evaluated only among genuinely proposed goals.
    candidate_dist2 = (pred.candidate_xy - target_xy[:, None]).square().sum(dim=-1)
    soft_target = torch.softmax(
        -candidate_dist2 / (2.0 * (1.5 / config.grid_size) ** 2), dim=-1
    )
    rank_loss = -(
        soft_target * F.log_softmax(pred.candidate_logits, dim=-1)
    ).sum(dim=-1).mean()
    losses["ranking"] = rank_loss
    total = total + ranking_weight * rank_loss

    if teacher_future is not None:
        expected = (
            target_xy.shape[0], config.trajectory_waypoints, 2
        )
        if teacher_future.shape != expected:
            raise ValueError(f"teacher_future must be {expected}")
        paths = pred.path_modes[batch_idx, nearest_idx]
        ade = (paths - teacher_future[:, None]).square().sum(-1).sqrt().mean(-1)
        path_loss = ade.amin(dim=-1).mean()
        losses["path"] = path_loss
        total = total + trajectory_weight * path_loss

    if physical_map_sizes is not None:
        if physical_map_sizes.shape != (target_xy.shape[0], 2):
            raise ValueError("physical_map_sizes must be [B,2] meters")
        dist_m = ((target_xy - uav_pose[:, :2]) * physical_map_sizes).norm(dim=-1)
        arriving = (dist_m <= stop_radius_m).float()
        current_stop = pred.stop_logits[batch_idx, nearest_idx]
        stop_loss = F.binary_cross_entropy_with_logits(current_stop, arriving)
        losses["stop"] = stop_loss
        total = total + stop_weight * stop_loss

    losses["total"] = total
    return losses
