from argparse import Namespace
import torch

from graphnav.config import GraphNavConfig
from graphnav.losses import navigation_losses
from graphnav.model import GraphNav
from graphnav.train import synthetic_minibatch


def test_synthetic_training_is_finite_and_updates_graph_parameters():
    torch.set_num_threads(2)
    cfg = GraphNavConfig(hidden_dim=32, heads=4, grid_size=4, max_landmarks=3,
                         contour_points=8, goal_top_k=3, trajectory_waypoints=3,
                         text_dim=16, vision_dim=6, graph_dropout=0)
    model = GraphNav(cfg)
    optim = torch.optim.AdamW(model.parameters(), lr=1e-3)
    batch, target = synthetic_minibatch(cfg, 2, torch.device("cpu"), 3)
    optim.zero_grad()
    result = model(**batch)
    losses = navigation_losses(result, target["target_xy"], batch["uav_pose"], cfg,
                               target["teacher_future"], target["physical_map_sizes"])
    losses["total"].backward()
    assert torch.isfinite(losses["total"])
    assert model.contour.point[0].weight.grad is not None
    assert model.static_graph.blocks[0].edge.weight.grad is not None
    optim.step()
