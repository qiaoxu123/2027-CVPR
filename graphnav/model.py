"""GraphNav: landmark graph -> dense goal field -> proposal graph -> trajectories.

No dependency on HETT, no GT/inference leakage, no full-city RGB input.
All positions (including polygon vertices) share normalized XY map coordinates.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import torch
from torch import nn
from torch.nn import functional as F

from .config import GraphNavConfig
from .geometry import (
    candidate_polygon_geometry, contour_statistics, grid_centers, node_edge_geometry
)
from .graph import EdgeGATv2, LandmarkGraphEncoder, spatial_adjacency


@dataclass
class GraphNavOutput:
    belief_logits: torch.Tensor           # [B, G*G], no visual evidence for unseen cells
    belief_probs: torch.Tensor            # [B, G*G]
    candidate_ids: torch.Tensor           # [B, K], selected from global grid
    candidate_xy: torch.Tensor            # [B, K, 2]
    candidate_logits: torch.Tensor        # [B, K], graph + masked visual correction
    path_modes: torch.Tensor              # [B, K, modes, steps, 2]
    path_logits: torch.Tensor             # [B, K, modes]
    stop_logits: torch.Tensor             # [B, K]
    candidate_hidden: torch.Tensor       # [B,K,H], permits vision-only cheap refinement
    selected_goal_xy: torch.Tensor        # [B, 2], inference-selected, no teacher goal
    next_waypoint_xy: torch.Tensor        # [B, 2], first waypoint of winning mode


class ContourEncoder(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.point = nn.Sequential(nn.Linear(2, hidden), nn.GELU(), nn.Linear(hidden, hidden))
        self.summary = nn.Sequential(nn.Linear(hidden * 2 + 8, hidden), nn.LayerNorm(hidden))

    def forward(self, contours: torch.Tensor):
        stats = contour_statistics(contours)
        rel = contours - stats[..., :2].unsqueeze(-2)
        points = self.point(rel)
        pooled = torch.cat((points.mean(dim=-2), points.amax(dim=-2), stats), dim=-1)
        return self.summary(pooled), stats[..., :2]


class DenseGoalField(nn.Module):
    """Full-grid differentiable candidate/landmark cross-attention.

    Scoring ALL cells prevents the top-K graph from losing unproposed goals.
    Polygon signed boundary distances are provided, including for line-like
    landmarks approximated as valid narrow polygons.
    """

    def __init__(self, hidden: int, dropout: float):
        super().__init__()
        self.position = nn.Sequential(nn.Linear(10, hidden), nn.GELU(), nn.Linear(hidden, hidden))
        self.anchor_geometry = nn.Sequential(
            nn.Linear(9, hidden), nn.GELU(), nn.Linear(hidden, hidden)
        )
        self.query = nn.Linear(hidden, hidden)
        self.key = nn.Linear(hidden, hidden)
        self.relation = nn.Linear(hidden, 1)
        self.null_anchor = nn.Parameter(torch.zeros(hidden))
        self.head = nn.Sequential(
            nn.LayerNorm(hidden * 3 + 2),
            nn.Linear(hidden * 3 + 2, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    @staticmethod
    def positional_features(xy):
        x, y = xy.unbind(dim=-1)
        pieces = [xy]
        for freq in (1.0, 2.0):
            pieces.extend((
                torch.sin(freq * 2.0 * torch.pi * x).unsqueeze(-1),
                torch.cos(freq * 2.0 * torch.pi * x).unsqueeze(-1),
                torch.sin(freq * 2.0 * torch.pi * y).unsqueeze(-1),
                torch.cos(freq * 2.0 * torch.pi * y).unsqueeze(-1),
            ))
        return torch.cat(pieces, dim=-1)

    def forward(self, xy, instruction, uav_node, landmarks, contours, landmark_valid, maps):
        b, q, _ = xy.shape
        query = self.position(self.positional_features(xy)) + instruction[:, None, :]
        rel = candidate_polygon_geometry(xy, contours)
        geom = self.anchor_geometry(rel)
        edge_scores = self.relation(
            torch.tanh(self.query(query)[:, :, None, :] + self.key(landmarks)[:, None] + geom)
        ).squeeze(-1)
        edge_scores = edge_scores.masked_fill(~landmark_valid[:, None, :], -1e4)
        null_scores = torch.zeros((b, q, 1), device=xy.device, dtype=xy.dtype)
        probabilities = torch.softmax(torch.cat((edge_scores, null_scores), dim=-1), dim=-1)
        contextualized_landmarks = landmarks[:, None] + geom
        null = self.null_anchor.view(1, 1, 1, -1).expand(b, q, 1, -1)
        values = torch.cat((contextualized_landmarks, null), dim=2)
        relevant = (probabilities.unsqueeze(-1) * values).sum(dim=2)
        logits = self.head(
            torch.cat((query, relevant, uav_node[:, None].expand(-1, q, -1), maps), dim=-1)
        ).squeeze(-1)
        return logits, query + relevant


class GraphNav(nn.Module):
    def __init__(self, config: GraphNavConfig = GraphNavConfig()):
        super().__init__()
        self.config = config
        h = config.hidden_dim
        self.text = nn.Sequential(nn.Linear(config.text_dim, h), nn.LayerNorm(h))
        self.name = nn.Sequential(nn.Linear(config.text_dim, h), nn.LayerNorm(h))
        self.contour = ContourEncoder(h)
        self.landmark = nn.Sequential(nn.Linear(h * 2, h), nn.GELU(), nn.LayerNorm(h))
        self.uav = nn.Sequential(nn.Linear(4, h), nn.GELU(), nn.LayerNorm(h))
        self.node_type = nn.Embedding(3, h)
        self.static_graph = LandmarkGraphEncoder(
            h, config.heads, config.graph_k_neighbors, config.graph_dropout
        )
        self.global_field = DenseGoalField(h, config.graph_dropout)
        self.goal_graph = EdgeGATv2(h, heads=config.heads, dropout=config.graph_dropout)
        self.goal_refine = nn.Sequential(nn.Linear(h, h), nn.GELU(), nn.Linear(h, 1))
        self.vision_proj = nn.Sequential(
            nn.Linear(config.vision_dim, h), nn.GELU(), nn.LayerNorm(h)
        )
        self.vision_score = nn.Sequential(nn.Linear(h * 2, h), nn.GELU(), nn.Linear(h, 1))
        # Two scalar residual gates safeguard new pretrained-free routing heads.
        self.graph_gate = nn.Parameter(torch.tensor(0.0))
        self.vision_gate = nn.Parameter(torch.tensor(0.0))
        self.trajectory = nn.Sequential(
            nn.Linear(h * 3, h), nn.GELU(),
            nn.Linear(h, config.trajectory_modes * config.trajectory_waypoints * 2),
        )
        self.mode_score = nn.Sequential(
            nn.Linear(h * 3, h), nn.GELU(), nn.Linear(h, config.trajectory_modes)
        )
        self.stop = nn.Sequential(nn.Linear(h * 3, h), nn.GELU(), nn.Linear(h, 1))
        self.register_buffer("grid", grid_centers(config.grid_size), persistent=False)

    def forward(
        self,
        instruction: torch.Tensor,
        landmark_names: torch.Tensor,
        landmark_contours: torch.Tensor,
        landmark_valid: torch.Tensor,
        uav_pose: torch.Tensor,
        spatial_maps: torch.Tensor | None = None,
        vision_features: torch.Tensor | None = None,
        vision_coverage: torch.Tensor | None = None,
        physical_map_sizes: torch.Tensor | None = None,
    ) -> GraphNavOutput:
        """No target coordinates are accepted. All arguments are inference-observable.

        instruction [B,T], names [B,M,T], polygons [B,M,P,2], mask [B,M],
        uav [B,4] = normalized x,y,sin(yaw),cos(yaw),
        maps [B,2,G,G] = current view and past explored masks,
        vision [B,K,V] and coverage [B,K] aligned to selected proposal order.
        physical_map_sizes [B,2] publicly known map widths/heights in meters.
        """
        c = self.config
        if instruction.ndim != 2 or instruction.shape[-1] != c.text_dim:
            raise ValueError("instruction shape must be [B,text_dim]")
        b = instruction.shape[0]
        if landmark_names.shape != (b, c.max_landmarks, c.text_dim):
            raise ValueError("landmark_names mismatch")
        if landmark_contours.shape != (b, c.max_landmarks, c.contour_points, 2):
            raise ValueError("landmark_contours mismatch")
        if landmark_valid.shape != (b, c.max_landmarks):
            raise ValueError("landmark_valid mismatch")
        if uav_pose.shape != (b, 4):
            raise ValueError("uav_pose mismatch")
        if spatial_maps is None:
            spatial_maps = instruction.new_zeros((b, 2, c.grid_size, c.grid_size))
        if spatial_maps.shape != (b, 2, c.grid_size, c.grid_size):
            raise ValueError("spatial_maps mismatch")
        if physical_map_sizes is not None and physical_map_sizes.shape != (b, 2):
            raise ValueError("physical_map_sizes must be public [B,2] map dimensions")
        if vision_features is not None and vision_coverage is None:
            raise ValueError("visual evidence requires explicit coverage mask")
        if vision_coverage is not None and vision_features is None:
            raise ValueError("coverage without evidence")
        if vision_features is not None:
            k_expected = min(c.goal_top_k, c.grid_size * c.grid_size)
            if vision_features.shape != (b, k_expected, c.vision_dim):
                raise ValueError("candidate vision shape/order mismatch")
            if vision_coverage.shape != (b, k_expected):
                raise ValueError("candidate coverage shape mismatch")

        reference = landmark_valid.bool()
        language = self.text(instruction)
        geom_emb, centers = self.contour(landmark_contours)
        lm = self.landmark(torch.cat((self.name(landmark_names), geom_emb), dim=-1))
        lm = (lm + self.node_type.weight[1]) * reference.unsqueeze(-1)
        uav_node = self.uav(uav_pose) + self.node_type.weight[0]
        graph_nodes = torch.cat((uav_node[:, None], lm), dim=1)
        graph_xy = torch.cat((uav_pose[:, None, :2], centers), dim=1)
        graph_valid = torch.cat(
            (torch.ones((b, 1), dtype=torch.bool, device=instruction.device), reference), dim=1
        )
        yaw = torch.atan2(uav_pose[:, 2], uav_pose[:, 3])
        graph_nodes = self.static_graph(graph_nodes, graph_xy, yaw, graph_valid)
        uav_context = graph_nodes[:, 0]
        lm_context = graph_nodes[:, 1:]

        q = self.grid.shape[0]
        candidate_xy = self.grid.to(instruction).unsqueeze(0).expand(b, -1, -1)
        grid_masks = spatial_maps.flatten(start_dim=2).transpose(1, 2)
        coarse_logits, grid_features = self.global_field(
            candidate_xy, language, uav_context, lm_context,
            landmark_contours, reference, grid_masks,
        )
        coarse_probs = torch.softmax(coarse_logits, dim=-1)
        topk = min(c.goal_top_k, q)
        candidate_ids = coarse_logits.topk(topk, dim=-1).indices
        candidate_xy = candidate_xy.gather(1, candidate_ids.unsqueeze(-1).expand(-1, -1, 2))
        candidate_feature = grid_features.gather(
            1, candidate_ids.unsqueeze(-1).expand(-1, -1, c.hidden_dim)
        ) + self.node_type.weight[2]

        # Separate proposal graph: proposals attend to all named landmarks and
        # the UAV even if those landmarks are geographically distant.
        xyz = torch.cat((graph_xy, candidate_xy), dim=1)
        valid = torch.cat(
            (graph_valid, torch.ones((b, topk), dtype=torch.bool, device=reference.device)),
            dim=1,
        )
        nodes = torch.cat((graph_nodes, candidate_feature), dim=1)
        edges = node_edge_geometry(xyz, yaw)
        polygon_edges = candidate_polygon_geometry(candidate_xy, landmark_contours)
        m = c.max_landmarks
        # Use the signed *polygon boundary* distance, not center distance.
        edges = edges.clone()
        edges[:, m + 1:, 1:m + 1, -1] = polygon_edges[..., 5]
        edges[:, 1:m + 1, m + 1:, -1] = polygon_edges[..., 5].transpose(1, 2)
        adjacency = spatial_adjacency(xyz, valid, c.graph_k_neighbors)
        pair = valid[:, :, None] & valid[:, None, :]
        adjacency[:, m + 1:, :m + 1] = pair[:, m + 1:, :m + 1]
        adjacency[:, :m + 1, m + 1:] = pair[:, :m + 1, m + 1:]
        proposal_graph = self.goal_graph(nodes, edges, adjacency, valid)[:, m + 1:]
        candidate_logits = coarse_logits.gather(1, candidate_ids) + (
            torch.tanh(self.graph_gate) * self.goal_refine(proposal_graph).squeeze(-1)
        )

        context = torch.cat(
            (proposal_graph, uav_context[:, None].expand(-1, topk, -1),
             language[:, None].expand(-1, topk, -1)),
            dim=-1,
        )
        offsets = torch.tanh(self.trajectory(context)).view(
            b, topk, c.trajectory_modes, c.trajectory_waypoints, 2
        ) * c.waypoint_residual_scale
        # Goal-conditioned paths: every proposed path terminates at its own
        # candidate. Intermediate waypoints carry learned geometric residuals.
        fractions = torch.linspace(
            1.0 / c.trajectory_waypoints, 1.0, c.trajectory_waypoints,
            device=instruction.device, dtype=instruction.dtype,
        )
        if physical_map_sizes is not None:
            # Replanning should make real metric progress: first waypoint is
            # at most 50m away, not 1/8 of an arbitrarily distant target.
            metric_delta = (
                candidate_xy - uav_pose[:, None, :2]
            ) * physical_map_sizes[:, None, :]
            distance_m = metric_delta.norm(dim=-1).clamp_min(1e-6)
            metric_fraction = (50.0 / distance_m).clamp(max=1.0)
            progress = torch.arange(
                1, c.trajectory_waypoints + 1,
                device=instruction.device, dtype=instruction.dtype,
            ).view(1, 1, 1, -1, 1)
            steps = (progress * metric_fraction[:, :, None, None, None]).clamp(max=1.0)
        else:
            # Geometry-only CPU tests may not have physical scene metadata;
            # production runtime always passes the public map dimensions.
            steps = fractions.view(1, 1, 1, -1, 1)
        base_path = (
            uav_pose[:, None, None, None, :2]
            + steps
            * (candidate_xy[:, :, None, None, :] - uav_pose[:, None, None, None, :2])
        )
        offsets = offsets.clone()
        offsets[..., -1, :] = 0.0
        path_modes = (base_path + offsets).clamp(0, 1)
        path_logits = self.mode_score(context)
        stop_logits = self.stop(context).squeeze(-1)
        scores = candidate_logits[:, :, None] + path_logits
        flattened = scores.reshape(b, -1).argmax(dim=-1)
        best_cand = torch.div(flattened, c.trajectory_modes, rounding_mode="floor")
        best_mode = flattened % c.trajectory_modes
        index = torch.arange(b, device=instruction.device)
        result = GraphNavOutput(
            belief_logits=coarse_logits,
            belief_probs=coarse_probs,
            candidate_ids=candidate_ids,
            candidate_xy=candidate_xy,
            candidate_logits=candidate_logits,
            path_modes=path_modes,
            path_logits=path_logits,
            stop_logits=stop_logits,
            candidate_hidden=proposal_graph,
            selected_goal_xy=candidate_xy[index, best_cand],
            next_waypoint_xy=path_modes[index, best_cand, best_mode, 0],
        )
        return (self.apply_visual_evidence(result, vision_features, vision_coverage)
                if vision_features is not None else result)

    def apply_visual_evidence(
        self, original: GraphNavOutput, vision_features: torch.Tensor,
        vision_coverage: torch.Tensor,
    ) -> GraphNavOutput:
        """Cheap second pass AFTER actual candidate RGB has been observed.

        Does NOT rerun the 784-cell polygon attention or the static landmark
        graph. A zero-coverage crop has exactly zero effect on candidate ranks.
        """
        b, k, h = original.candidate_hidden.shape
        if vision_features.shape != (b, k, self.config.vision_dim):
            raise ValueError("vision feature dimensions or proposal ordering mismatch")
        if vision_coverage.shape != (b, k):
            raise ValueError("vision coverage dimensions mismatch")
        trusted = (vision_coverage >= self.config.min_visual_coverage).float()
        visual = self.vision_proj(vision_features)
        adjustment = self.vision_score(
            torch.cat((original.candidate_hidden, visual), dim=-1)
        ).squeeze(-1) * trusted * torch.tanh(self.vision_gate)
        logits = original.candidate_logits + adjustment
        # Keep a coherent UPDATED global belief. Unseen cells are unchanged;
        # only proposals with new trusted observations receive a residual.
        updated_field = original.belief_logits.scatter_add(
            1, original.candidate_ids, adjustment
        )
        total_scores = logits[:, :, None] + original.path_logits
        flattened = total_scores.reshape(b, -1).argmax(-1)
        candidate_id = torch.div(
            flattened, self.config.trajectory_modes, rounding_mode="floor"
        )
        mode_id = flattened % self.config.trajectory_modes
        batch = torch.arange(b, device=logits.device)
        return replace(
            original,
            belief_logits=updated_field,
            belief_probs=torch.softmax(updated_field, dim=-1),
            candidate_logits=logits,
            selected_goal_xy=original.candidate_xy[batch, candidate_id],
            next_waypoint_xy=original.path_modes[batch, candidate_id, mode_id, 0],
        )


def bounded_next_step(current_world, next_world, max_step_m: float = 50.0):
    """Limit a planned waypoint in physical XY meters; never use GT."""
    delta = next_world - current_world
    norm = delta.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    return current_world + delta * (max_step_m / norm).clamp(max=1.0)
