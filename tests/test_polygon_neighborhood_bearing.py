"""CPU contracts for GT-free public Polygon neighborhood bearing study."""
import numpy as np
import torch

from graphnav.polygon_neighborhood_bearing import (
    NeighborhoodBearingHead, PublicPolygonNeighborhood,
)
from scripts.train_polygon_neighborhood_bearing import (
    BearingExamples, angular_loss, choose_maps, collate,
)


def square(x, y, radius=2.0):
    return [[x-radius, y-radius], [x+radius, y-radius],
            [x+radius, y+radius], [x-radius, y+radius]]


def catalog():
    return {"map_0": {
        "a": {"name": "Anchor", "object_type": "Building", "contour": square(50, 50)},
        "right": {"name": "", "object_type": "Building", "contour": square(70, 50)},
        "up": {"name": "", "object_type": "Parking", "contour": square(50, 65)},
        "far": {"name": "", "object_type": "Car", "contour": square(300, 300)},
    }}


def record(target_xy=(.7, .5)):
    return dict(map_name="map_0", instruction="the building east of Anchor",
                landmarks=[dict(id="a", name="Anchor",
                                object_type="Building", contour=square(.5, .5, .02))],
                world_bounds=[0, 0, 100, 100],
                target_xy=list(target_xy))


def test_public_neighbors_selected_by_anchor_not_goal():
    builder = PublicPolygonNeighborhood(catalog(), radius_m=50., max_refs=2,
                                        max_neighbors=4)
    one = builder.for_references("map_0", [{"id": "a"}])
    np.testing.assert_allclose(one["anchor_centers_world"][0], [50, 50])
    assert one["landmark_valid"].tolist() == [True, False]
    assert one["neighbor_valid"][0].sum() == 2  # far is not selected
    # Closest is the object 15m above the anchor.
    assert one["neighbor_geometry"][0, 0, 1] > 0
    assert one["neighbor_geometry"][0, 1, 0] > 0
    assert not one["neighbor_valid"][1].any()
    assert np.all(one["anchor_centers_world"][1] == 0)


def test_padded_unknown_reference_is_not_replaced_by_nearest_object():
    builder = PublicPolygonNeighborhood(catalog(), radius_m=50., max_refs=2,
                                        max_neighbors=3)
    geo = builder.for_references("map_0", [{"id": "not-in-public-catalog"}])
    assert not geo["landmark_valid"].any()
    assert not geo["neighbor_valid"].any()
    assert not geo["anchor_centers_world"].any()


def test_feature_selection_does_not_read_target_gt():
    row = record()
    dataset = BearingExamples([row], catalog(), max_refs=2, neighbors=3)
    a = dataset[0]
    row["target_xy"] = [.5, .9]
    b = dataset[0]
    for key in ("anchor_geometry", "neighbor_geometry", "neighbor_types",
                "neighbor_valid", "landmark_valid", "anchor_centers_world"):
        # anchor_centers_world is internal to the geometry builder, not output.
        if key in a:
            np.testing.assert_array_equal(a[key], b[key])
    assert a["labels"][0] == 0
    assert b["labels"][0] == 2


def test_direction_head_geometry_ablation_and_gradients():
    torch.manual_seed(42)
    head = NeighborhoodBearingHead(text_dim=8, hidden=16, bins=8)
    x = dict(tokens=torch.randn(2, 5, 8),
             token_mask=torch.tensor([[True]*5, [True]*4+[False]),
             node_emb=torch.randn(2, 2, 8),
             landmark_valid=torch.tensor([[True, True], [True, False]]),
             anchor_geometry=torch.rand(2, 2, 5),
             neighbor_geometry=torch.rand(2, 2, 3, 8),
             neighbor_types=torch.tensor([[[1, 2, 3], [1, 1, 0]],
                                          [[3, 2, 0], [0, 0, 0]]]),
             neighbor_valid=torch.tensor([[[True, True, True], [True, False, False]],
                                          [[True, True, False], [False, False, False]]]))
    y = head(**x)
    assert y.shape == (2, 2, 8)
    assert torch.equal(y[1, 1], torch.zeros(8))
    changed = dict(x)
    changed["neighbor_geometry"] = x["neighbor_geometry"].clone()
    changed["neighbor_geometry"][0, 0, 0, 0] += .8
    assert not torch.allclose(head(**x)[0, 0], head(**changed)[0, 0])
    torch.testing.assert_close(head(**x, geometry_enabled=False),
                               head(**changed, geometry_enabled=False))
    loss = torch.nn.functional.cross_entropy(y[0, 0][None], torch.tensor([1]))
    loss.backward()
    assert head.neighbor.weight.grad is not None
    assert head.neighbor.weight.grad.abs().sum() > 0


def test_equal_query_weight_and_no_label():
    logits = torch.zeros(2, 3, 8, requires_grad=True)
    label = torch.tensor([[0, 1, 2], [3, -100, -100]])
    loss, n, nq = angular_loss(logits, label)
    assert n == 4 and nq == 2
    assert abs(float(loss) - np.log(8)) < 1e-6
    loss.backward()
    assert logits.grad is not None
    zero, count, questions = angular_loss(logits, torch.full_like(label, -100))
    assert zero.item() == 0 and count == 0 and questions == 0


def test_map_split_blocks_same_map_in_both_sets():
    records = [dict(map_name=f"m{i%5}") for i in range(60)]
    train, held, names = choose_maps(records)
    assert train and held and names
    assert not ({records[i]["map_name"] for i in train} &
                {records[i]["map_name"] for i in held})
    assert sorted(train + held) == list(range(60))
    assert choose_maps(records)[2] == names


def test_collate_preserves_reference_and_label_alignment():
    ds = BearingExamples([record()], catalog(), max_refs=2, neighbors=3)
    batch = collate([ds[0], ds[0]])
    assert batch["neighbor_geometry"].shape == (2, 2, 3, 8)
    assert batch["labels"].tolist() == [[0, -100], [0, -100]]
    assert batch["n_refs"].tolist() == [1, 1]
