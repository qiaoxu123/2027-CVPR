"""CPU invariants for the joint RGB/geometry *experimental* arm."""
import json

import torch

from graphnav.candidate_geometry import FullCandidateDataset, collate_full
from graphnav.full_localization import FullTokenGeometryLocalizer
from graphnav.joint_rgb_geometry import JointRGBGeometryStage1


def inputs(tmp_path):
    data = dict(
        map_name='m', instruction='red car east of Church',
        world_bounds=[0, 0, 400, 400], physical_map_sizes=[400, 400],
        target_xy=[.7, .5],
        landmarks=[dict(name='Church', object_type='Building',
                        contour=[[.1,.1],[.2,.1],[.2,.2],[.1,.2]])],
    )
    path = tmp_path/'queries.jsonl'
    path.write_text(json.dumps(data)+'\n')
    static = collate_full([FullCandidateDataset(
        path, grid_size=4, max_landmarks=2,
        max_roads=0, max_relations=2,
    )[0]])
    a3 = FullTokenGeometryLocalizer(
        hidden=16, text_dim=8, heads=4, max_relations=2,
        top_k=8, vision_dim=8, dropout=0,
    )
    model = JointRGBGeometryStage1(a3, vision_dim=8, matcher_hidden=4)
    model.configure_stage('C')
    model.eval()
    x = dict(
        static_inputs=static,
        tokens=torch.randn(1, 5, 8),
        token_mask=torch.ones(1, 5, dtype=torch.bool),
        node_emb=torch.randn(1, static['nodes'].shape[1], 8),
        native_text=torch.randn(1, 8),
        appearance_text=torch.randn(1, 8),
        roi_features=torch.randn(1, 16, 3, 8),
        roi_valid=torch.zeros(1, 16, 3, dtype=torch.bool),
        roi_quality=torch.ones(1, 16, 3, 3),
        pose_world=torch.tensor([[200., 200., 50., 0.]]),
        bounds=torch.tensor([[0., 0., 400., 400.]]),
        ground_level=torch.zeros(1),
    )
    return model, x


def test_all_unknown_is_exact_original_field(tmp_path):
    torch.manual_seed(7)
    model, x = inputs(tmp_path)
    with torch.no_grad():
        out = model(**x)
    assert out.belief is out.static
    assert out.delta.count_nonzero() == 0
    assert torch.equal(out.belief.candidate_ids, out.static.candidate_ids)
    assert torch.equal(out.belief.candidate_context, out.static.candidate_context)


def test_zero_rgb_keeps_static_scores_but_nonzero_rgb_trains_joint_fusion(tmp_path):
    torch.manual_seed(7)
    model, x = inputs(tmp_path)
    x['roi_valid'][:, 0] = True
    with torch.no_grad():
        zero = model(**dict(x, visual_control='zero'))
    torch.testing.assert_close(zero.belief.belief_logits, zero.static.belief_logits, rtol=0, atol=0)
    torch.testing.assert_close(zero.belief.dense_context, zero.static.dense_context, rtol=0, atol=0)
    out = model(**x)
    assert not torch.equal(out.belief.belief_logits[:, 0], out.static.belief_logits[:, 0])
    torch.testing.assert_close(out.belief.belief_logits[:, 1:], out.static.belief_logits[:, 1:], rtol=0, atol=0)
    torch.testing.assert_close(out.belief.dense_context[:, 1:], out.static.dense_context[:, 1:], rtol=0, atol=0)
    out.belief.belief_logits.sum().backward()
    grad = model.a3.rgb_query[1].weight.grad
    assert grad is not None and grad.abs().sum() > 0


def test_unknown_pixel_features_cannot_change_goal(tmp_path):
    torch.manual_seed(9)
    model, x = inputs(tmp_path)
    x['roi_valid'][:, 0] = True
    with torch.no_grad():
        first = model(**x)
        y = dict(x)
        altered = x['roi_features'].clone()
        altered[:, 1:] *= 1e3
        y['roi_features'] = altered
        second = model(**y)
    torch.testing.assert_close(first.belief.belief_logits, second.belief.belief_logits, rtol=0, atol=0)
    torch.testing.assert_close(first.belief.dense_context, second.belief.dense_context, rtol=0, atol=0)
