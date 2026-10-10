"""Small dense edge-conditioned GATv2; no torch-geometric dependency.

Legacy navigation graph: 1 UAV + at most 16 landmarks + at most 20 proposals.

IMPORTANT: the static-first experiment uses semantic_graph.py, not this
legacy UAV-to-all/k-NN adjacency. This remains for backward compatibility.
Quadratic attention is limited to this small graph, NOT all map cells.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .geometry import node_edge_geometry


def spatial_adjacency(xy: torch.Tensor, valid: torch.Tensor, k: int) -> torch.Tensor:
    """Directed k-NN graph; UAV (node zero) links to all named landmarks."""
    b, n, _ = xy.shape
    distance = torch.cdist(xy, xy)
    pair = valid[:, :, None] & valid[:, None, :]
    distance = distance.masked_fill(~pair, float("inf"))
    idx = distance.topk(k=min(max(k, 1), n), dim=-1, largest=False).indices
    adj = torch.zeros_like(distance, dtype=torch.bool)
    adj.scatter_(-1, idx, True)
    adj &= pair
    adj[:, 0, :] = pair[:, 0, :]
    adj[:, :, 0] = pair[:, :, 0]
    eye = torch.eye(n, dtype=torch.bool, device=xy.device).unsqueeze(0)
    return adj | (eye & pair)


class EdgeGATv2(nn.Module):
    """Multihead edge-conditioned dynamic attention + geometric value messages."""

    def __init__(self, hidden: int, edge_dim: int = 8, heads: int = 4, dropout: float = 0.1):
        super().__init__()
        if hidden % heads:
            raise ValueError("hidden must be divisible by heads")
        self.heads = heads
        self.width = hidden // heads
        self.src = nn.Linear(hidden, hidden, bias=False)
        self.dst = nn.Linear(hidden, hidden, bias=False)
        self.edge = nn.Linear(edge_dim, hidden, bias=False)
        self.score = nn.Parameter(torch.randn(heads, self.width) * 0.02)
        self.value = nn.Linear(hidden, hidden, bias=False)
        self.edge_value = nn.Linear(edge_dim, hidden, bias=False)
        self.out = nn.Linear(hidden, hidden)
        self.norm = nn.LayerNorm(hidden)
        self.dropout = nn.Dropout(dropout)

    def forward(self, nodes, edges, adjacency, valid):
        b, n, hidden = nodes.shape
        q = self.src(nodes).view(b, n, self.heads, self.width)
        k = self.dst(nodes).view(b, n, self.heads, self.width)
        e = self.edge(edges).view(b, n, n, self.heads, self.width)
        z = F.leaky_relu(q[:, :, None] + k[:, None] + e, negative_slope=0.2)
        scores = (z * self.score).sum(-1)
        scores = scores.masked_fill(~adjacency.unsqueeze(-1), torch.finfo(scores.dtype).min)
        attention = self.dropout(torch.softmax(scores, dim=2))
        v = self.value(nodes).view(b, n, self.heads, self.width)
        ev = self.edge_value(edges).view(b, n, n, self.heads, self.width)
        aggregated = (attention.unsqueeze(-1) * (v[:, None] + ev)).sum(dim=2)
        output = self.out(aggregated.reshape(b, n, hidden))
        return self.norm(nodes + output) * valid.unsqueeze(-1)


class LandmarkGraphEncoder(nn.Module):
    """Graph message passing between the UAV and named landmark polygons."""

    def __init__(self, hidden: int, heads: int, k_neighbors: int, dropout: float):
        super().__init__()
        self.k_neighbors = k_neighbors
        self.blocks = nn.ModuleList(
            [EdgeGATv2(hidden, heads=heads, dropout=dropout) for _ in range(2)]
        )

    def forward(self, nodes, xy, yaw, valid):
        edges = node_edge_geometry(xy, yaw)
        adjacency = spatial_adjacency(xy, valid, self.k_neighbors)
        for block in self.blocks:
            nodes = block(nodes, edges, adjacency, valid)
        return nodes
