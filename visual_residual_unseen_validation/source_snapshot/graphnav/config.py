"""Explicit, checkpoint-serializable model hyperparameters."""

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class GraphNavConfig:
    text_dim: int = 768
    vision_dim: int = 768
    hidden_dim: int = 128
    heads: int = 4
    grid_size: int = 28
    max_landmarks: int = 16
    contour_points: int = 16
    goal_top_k: int = 20
    trajectory_modes: int = 3
    trajectory_waypoints: int = 8
    graph_k_neighbors: int = 4
    graph_dropout: float = 0.1
    waypoint_residual_scale: float = 0.025
    # Normalize all positions to the same fixed, public map bounds [0, 1]^2.
    # Physical world distances are computed using the bounds, not a presumed 400 m map.
    min_visual_coverage: float = 0.30

    def __post_init__(self) -> None:
        if self.hidden_dim % self.heads != 0:
            raise ValueError("hidden_dim must be divisible by heads")
        for name in (
            "grid_size", "max_landmarks", "contour_points",
            "goal_top_k", "trajectory_modes", "trajectory_waypoints",
        ):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        if not 0 <= self.min_visual_coverage <= 1:
            raise ValueError("min_visual_coverage must be in [0,1]")

    def to_dict(self) -> dict:
        return asdict(self)
