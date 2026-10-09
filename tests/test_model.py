import torch

from graphnav.config import GraphNavConfig
from graphnav.model import GraphNav, bounded_next_step
from graphnav.losses import navigation_losses


def sample(cfg, batch=2):
    torch.manual_seed(4)
    polygons = torch.rand(batch, cfg.max_landmarks, cfg.contour_points, 2)
    center = torch.rand(batch, cfg.max_landmarks, 1, 2) * 0.7
    # Intentionally ordered, non-self-crossing square-like polygons.
    theta = torch.linspace(0, 2 * torch.pi, cfg.contour_points + 1)[:-1]
    offsets = torch.stack((torch.cos(theta), torch.sin(theta)), dim=-1)
    polygons = center + offsets[None, None] * 0.03
    return dict(
        instruction=torch.randn(batch, cfg.text_dim),
        landmark_names=torch.randn(batch, cfg.max_landmarks, cfg.text_dim),
        landmark_contours=polygons,
        landmark_valid=torch.tensor([[True, False, True] for _ in range(batch)]),
        uav_pose=torch.tensor([[0.1, 0.3, 0.0, 1.0] for _ in range(batch)]),
        spatial_maps=torch.zeros(batch, 2, cfg.grid_size, cfg.grid_size),
    )


def test_forward_shapes_goal_and_waypoints():
    cfg = GraphNavConfig(hidden_dim=32, heads=4, grid_size=5, max_landmarks=3,
                         contour_points=8, goal_top_k=4, trajectory_waypoints=3)
    model = GraphNav(cfg)
    out = model(**sample(cfg))
    assert out.belief_logits.shape == (2, 25)
    assert out.path_modes.shape == (2, 4, 3, 3, 2)
    assert out.stop_logits.shape == (2, 4)
    assert out.next_waypoint_xy.shape == (2, 2)
    assert torch.allclose(out.belief_probs.sum(dim=-1), torch.ones(2))
    assert torch.isfinite(out.belief_logits).all()


def test_observation_mask_prevents_visual_leakage():
    cfg = GraphNavConfig(hidden_dim=32, heads=4, grid_size=4, max_landmarks=3,
                         contour_points=8, goal_top_k=4, vision_dim=6, graph_dropout=0)
    model = GraphNav(cfg).eval()
    with torch.no_grad():
        model.vision_gate.fill_(1.0)
    batch = sample(cfg)
    features = torch.randn(2, 4, 6)
    unseen = torch.zeros(2, 4)
    one = model(**batch, vision_features=features, vision_coverage=unseen)
    two = model(**batch, vision_features=features * 1000, vision_coverage=unseen)
    assert torch.allclose(one.candidate_logits, two.candidate_logits, atol=1e-6)
    assert torch.allclose(one.belief_logits, two.belief_logits, atol=1e-6)
    partial = model(**batch, vision_features=features, vision_coverage=torch.ones(2, 4))
    assert not torch.allclose(one.candidate_logits, partial.candidate_logits)


def test_loss_backward_and_physical_step():
    cfg = GraphNavConfig(hidden_dim=32, heads=4, grid_size=4, max_landmarks=3,
                         contour_points=8, goal_top_k=4, trajectory_waypoints=3)
    model = GraphNav(cfg)
    batch = sample(cfg)
    output = model(**batch)
    goal = torch.tensor([[0.7, 0.8], [0.6, 0.3]])
    path = torch.stack([
        batch["uav_pose"][:, :2] * (1 - alpha) + goal * alpha
        for alpha in (1 / 3, 2 / 3, 1)
    ], dim=1)
    loss = navigation_losses(output, goal, batch["uav_pose"], cfg, path,
                             torch.tensor([[400.0, 400.0], [400.0, 400.0]]))
    loss["total"].backward()
    assert torch.isfinite(loss["total"])
    assert model.global_field.head[-1].weight.grad is not None
    step = bounded_next_step(torch.tensor([[0.0, 0.0]]),
                             torch.tensor([[100.0, 0.0]]), 50.0)
    assert torch.allclose(step, torch.tensor([[50.0, 0.0]]))
