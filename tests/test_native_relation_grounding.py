"""CPU-only contract tests for the CityNav relation-grounding prototype."""
import inspect

import numpy as np
import torch

from graphnav.native_relation_grounding import (
    RelationAwareQueryRanker, public_relation_descriptors,
    weak_relation_ranking_loss,
)
from graphnav.semantic_geometry import SpatialRelation
from scripts.train_native_relational_grounding import (
    augment_public_relations, collate_relation,
)


def graph(kind='north', ambiguous=False):
    entity = dict(kind='building', name='The School',
                  geometry_m=np.array([
                      [40., 40.], [60., 40.],
                      [60., 60.], [40., 60.],
                  ], dtype=np.float32))
    entities = [entity]
    if ambiguous:
        entities.append(dict(entity))
    return dict(entities=entities, relations=[
        SpatialRelation(kind, ((0, 1) if ambiguous else (0,),)),
    ])


def batch(kind='north', ambiguous=False):
    xy = torch.tensor([[50., 90.], [50., 10.], [10., 50.]])
    f, t, m = public_relation_descriptors(xy.numpy(), graph(kind, ambiguous))
    features = torch.randn(1, 3, 256)
    text = torch.randn(1, 4, 256)
    text_mask = torch.tensor([[True, True, True, False]])
    geo = torch.zeros(1, 3, 8)
    valid = torch.tensor([[True, True, False]])
    scores = torch.tensor([[1.0, 1.0, -1e4]])
    null = torch.tensor([-.5])
    return dict(
        features=features, text=text, text_mask=text_mask,
        geometry=geo, valid=valid, relation_features=f[None],
        relation_types=t[None], relation_valid=m[None],
        native_logits=scores, native_null=null,
    )


def test_public_north_relation_world_sign():
    x = batch()
    f = x['relation_features'][0]
    mask = x['relation_valid'][0]
    assert mask.sum().item() == 1
    assert f[0, 0, 2] > 0 and f[1, 0, 2] < 0
    assert f[0, 0, 0] > f[1, 0, 0]


def test_unmatched_and_duplicate_reference_no_relation_fallback():
    torch.manual_seed(17)
    model = RelationAwareQueryRanker()
    with torch.no_grad():
        model.head[-1].bias.fill_(2.)
    x = batch(ambiguous=True)
    out = model(**x)
    assert not out['has_relation'].any()
    assert torch.equal(out['logits'], x['native_logits'])
    assert torch.equal(out['null'], x['native_null'])
    assert not out['attention'].any()


def test_zero_initialized_head_starts_from_frozen_native_scores():
    x = batch()
    model = RelationAwareQueryRanker()
    out = model(**x)
    assert torch.equal(out['logits'], x['native_logits'])
    assert torch.equal(out['null'], x['native_null'])
    assert out['attention'][0, 0].sum().item() == 1.


def test_relation_disabled_always_exact_native_and_masked_candidate_ignored():
    x = batch()
    model = RelationAwareQueryRanker()
    with torch.no_grad():
        model.head[-1].bias.fill_(4.)
    out = model(**x, relation_enabled=False)
    assert torch.equal(out['logits'], x['native_logits'])
    assert not out['has_relation'].any()
    x['valid'][0, 2] = False
    out = model(**x)
    assert out['logits'][0, 2] == -1e4


def test_relation_training_gradients_only_when_supervised():
    x = batch()
    model = RelationAwareQueryRanker()
    scores = model(**x)
    target = torch.tensor([[50., 90.]])
    centers = torch.tensor([[[50., 90.], [50., 10.], [10., 50.]]])
    loss, stats = weak_relation_ranking_loss(
        scores['logits'], x['valid'], scores['has_relation'],
        centers, target)
    assert stats['supervised_rows'] == 1
    loss.backward()
    assert model.head[-1].weight.grad is not None
    assert model.head[-1].weight.grad.abs().sum() > 0


def test_unobserved_target_not_treated_as_negative_instance():
    x = batch()
    model = RelationAwareQueryRanker()
    scores = model(**x)
    centers = torch.tensor([[[50., 90.], [50., 10.], [10., 50.]]])
    loss, stats = weak_relation_ranking_loss(
        scores['logits'], x['valid'], scores['has_relation'],
        centers, torch.tensor([[500., 900.]]))
    assert stats['supervised_rows'] == 0
    assert torch.isfinite(loss)
    assert loss == 0


def test_no_private_label_in_relation_forward():
    signature = inspect.signature(RelationAwareQueryRanker.forward)
    assert 'target_world' not in signature.parameters
    signature = inspect.signature(public_relation_descriptors)
    assert 'target_world' not in signature.parameters


def test_bad_geometry_and_mask_shape_rejected():
    x = batch()
    model = RelationAwareQueryRanker()
    x['relation_features'] = torch.zeros(1, 3, 9, 8)
    try:
        model(**x)
    except ValueError:
        return
    raise AssertionError('Invalid reference tensor must not pass')


def test_public_augment_uses_only_map_instruction_and_query_xy():
    class Static:
        records = [dict(map_name='birmingham', instruction='north of the school')]
        def _prepared(self, index):
            assert index == 0
            return graph(), None
    class Dataset:
        static = Static()
    row = dict(map_name='birmingham', instruction='north of the school',
               query_world=torch.tensor([[50., 90.], [50., 10.], [0., 0.]]),
               labels=dict(target_world=torch.tensor([50., 90.])))
    prepared = augment_public_relations([row], Dataset())
    assert 'relation_features' in prepared[0]
    assert 'relation_features' not in row
    # Inverting GT must not change public graph features.
    fake = dict(row, labels=dict(target_world=torch.tensor([500., 900.])))
    other = augment_public_relations([fake], Dataset())
    assert torch.equal(other[0]['relation_features'], prepared[0]['relation_features'])


def test_explicit_ambiguity_and_empty_relations():
    x = graph()
    x['relations'] = []
    f, t, mask = public_relation_descriptors(
        np.array([[10., 20.]], np.float32), x)
    assert f.shape == (1, 8, 8)
    assert not mask.any()
    assert not f.any()


def test_diagonal_world_relation_is_not_silently_reduced_to_east():
    from graphnav.semantic_geometry import parse_spatial_relations
    entities=graph()['entities']
    for phrase in ['north-east of The School','north east of The School']:
        parsed=parse_spatial_relations(phrase,entities)
        assert [r.kind for r in parsed]==['northeast']


def test_masked_tokens_are_ignored_and_active_nonfinite_text_fails_closed():
    import pytest
    x=batch();model=RelationAwareQueryRanker()
    with torch.no_grad():model.head[-1].weight.fill_(.05)
    expected=model(**x)['logits']
    x['text'][0,-1]=float('nan')
    assert torch.equal(model(**x)['logits'],expected)
    x['text'][0,0]=float('nan')
    with pytest.raises(ValueError,match='Nonfinite'):model(**x)


def test_masked_reference_cannot_change_attention_or_scores():
    x=batch();model=RelationAwareQueryRanker()
    with torch.no_grad():model.head[-1].weight.normal_(0,.05)
    expected=model(**x)
    x['relation_features'][:,:,1:]=10000
    actual=model(**x)
    assert torch.equal(actual['logits'],expected['logits'])
    assert not actual['attention'][:,:,1:].any()
    assert not actual['attention'][0,2].any()


def test_relation_geometry_uses_translation_invariant_world_coordinates():
    g=graph();xy=np.array([[50.,90.],[50.,10.]],np.float32)
    original=public_relation_descriptors(xy,g)[0]
    offset=np.array([10000.,-3000.],np.float32)
    shifted=graph();shifted['entities'][0]['geometry_m']+=offset
    actual=public_relation_descriptors(xy+offset,shifted)[0]
    assert torch.allclose(actual,original,atol=1e-6)


def test_multi_reference_attention_backpropagates_to_query_and_key():
    x = batch()
    x['relation_valid'][0, 1] = True
    x['relation_features'][:, :, 1] = -x['relation_features'][:, :, 0]
    x['relation_types'][0, 1] = 1
    model = RelationAwareQueryRanker()
    # The zero-initialized output head intentionally blocks upstream gradients
    # at initialization; verify the attention path once the head is active.
    with torch.no_grad():
        model.head[-1].weight.normal_(0., .1)
    model(**x)['logits'][0, 0].backward()
    for layer in [model.query, model.key, model.reference[0]]:
        assert layer.weight.grad is not None
        assert torch.isfinite(layer.weight.grad).all()
        assert layer.weight.grad.abs().sum() > 0


def test_shuffled_ablation_changes_only_valid_query_relation_geometry():
    from scripts.train_native_relational_grounding import shuffled_relation_inputs
    x = batch()
    changed = shuffled_relation_inputs(x)
    assert torch.equal(changed['relation_features'][0, 0], x['relation_features'][0, 1])
    assert torch.equal(changed['relation_features'][0, 2], x['relation_features'][0, 2])
    for key in x.keys() - {'relation_features'}:
        assert torch.equal(changed[key], x[key])
