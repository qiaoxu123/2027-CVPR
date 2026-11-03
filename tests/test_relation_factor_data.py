import inspect
import torch
from graphnav.relation_factor_data import public_reference_inputs


def build(role):
    return public_reference_inputs(world_hypotheses=torch.tensor([[0., 0.], [1., 1.], [100., 100.]]),
        global_cells=1, quality=torch.ones(2, 6)*.001, role=role,
        text_vectors=torch.ones(len(role['references']), 8),
        map_objects={'1': {'contour': [[0., 0.], [2., 0.], [2., 2.], [0., 2.]]}}, capacity=2)


def test_named_support_respects_public_polygon_and_ordinary_is_not_masked():
    ref = dict(phrase='Foo', kind='near', public_ids=['1'],
               anchors=[dict(kind='identity', public_ids=['1'])])
    out = build({'references': [ref]})
    assert out['reference_support'][0].tolist() == [True, False]
    ref['public_ids'] = []
    assert build({'references': [ref]})['reference_support'][0].all()


def test_padding_is_inactive_and_api_has_no_private_truth():
    out = build({'references': []})
    assert not out['reference_valid'].any()
    assert not out['reference_support'].any()
    assert not {'target_world', 'target_id', 'contour'} & set(inspect.signature(public_reference_inputs).parameters)
