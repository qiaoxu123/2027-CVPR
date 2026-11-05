import torch
import pytest
from graphnav.direct_target import DirectTargetHead, project_patch_evidence


def test_observation_support_and_no_label_forward():
    uv = torch.tensor([[[.25, .25], [.75, .75]]])
    scores = torch.tensor([[2., -1.]], requires_grad=True)
    delta, cover = project_patch_evidence(uv, scores, torch.tensor([[True, False]]), 4)
    assert torch.equal(delta[~cover], torch.zeros_like(delta[~cover]))
    delta.sum().backward()
    assert scores.grad[0, 0] != 0 and scores.grad[0, 1] == 0
    with pytest.raises(TypeError):
        project_patch_evidence(uv, scores, torch.ones(1, 2, dtype=torch.bool), 4, target_xy=[0, 0])


def test_relation_text_gradient_and_mask():
    torch.manual_seed(42)
    head = DirectTargetHead(hidden=16, text_dim=8, visual_dim=12)
    dense = torch.randn(2, 16, 16, requires_grad=True)
    tokens = torch.randn(2, 5, 8, requires_grad=True)
    mask = torch.ones(2, 5, dtype=torch.bool)
    mask[:, -1] = False
    geo = torch.randn(2, 1, 16, 10)
    ids = torch.tensor([[0], [1]])
    valid = torch.ones(2, 1, dtype=torch.bool)
    out = head.relation(dense, tokens, mask, geo, ids, valid)
    changed = tokens.detach().clone(); changed[:, -1] = 1e6
    assert torch.allclose(out, head.relation(dense.detach(), changed, mask, geo, ids, valid))
    inverse = head.relation(dense, tokens, mask, geo, 1-ids, valid)
    assert not torch.allclose(out, inverse)
    (out * torch.randn_like(out)).sum().backward()
    assert head.word.weight.grad.abs().sum() > 0
    assert head.predicate.weight.grad.abs().sum() > 0
    assert dense.grad.abs().sum() > 0
    assert tokens.grad[:, -1].abs().sum() == 0


def test_missing_rgb_exact_fallback():
    head = DirectTargetHead(hidden=16, text_dim=8, visual_dim=12)
    logits = torch.randn(2, 16)
    actual, coverage = head.visual(logits, None, None, None, None, None)
    assert torch.equal(logits, actual)
    assert not coverage.any()


def test_direct_field_readiness_and_candidate_alignment():
    from tests.test_unified_pgvb_v2 import TinyGeometry, _inputs
    from graphnav.unified_pgvb import UnifiedPGVB
    model = UnifiedPGVB(geometry=TinyGeometry(), direct_target=True)
    inputs = _inputs(); inputs.update(tokens=torch.randn(1, 5, 768),
        token_mask=torch.ones(1, 5, dtype=torch.bool),
        relation_geometry=torch.randn(1, 1, 784, 10),
        relation_ids=torch.zeros(1, 1, dtype=torch.long),
        relation_valid=torch.ones(1, 1, dtype=torch.bool))
    result = model.forward_direct_target(**inputs)
    field = result['field']; ids = field.candidate_ids
    assert torch.equal(ids, field.belief_logits.topk(20, dim=-1).indices)
    assert torch.equal(field.candidate_xy, inputs['static_inputs']['grid_xy'].gather(1, ids[..., None].expand(-1,-1,2)))
    assert torch.equal(field.candidate_context, field.dense_context.gather(1, ids[...,None].expand(-1,-1,128)))
    model.eval()
    with pytest.raises(RuntimeError, match='Untrained'):
        model.forward_direct_target(**inputs)


def test_public_language_selection_rejects_indirect_and_relative():
    from graphnav.direct_target_language import select_direct_relations
    obj={'1':dict(name='Museum',object_type='Building',contour=[[0,0],[1,0],[1,1],[0,1]])}
    def parsed(text, phrase, start):
        return dict(original_text=text,target_GT_used=False,ambiguity=[],
            target=dict(entity='t',category='Car'),
            entities=[dict(id='t',span=[0,3],text='Car')],
            landmarks=[dict(public_names=['Museum'],start=start,end=start+6)],
            relations=[dict(id='r',subject='t',object='l',predicate=phrase,
                relation_kind='spatial_proposal',ambiguity=[],source_span=[0,len(text)],
                source_text=text,reference_frame=dict(frame='metric_or_adjacency'))],
            direct_relations=[dict(landmark_names=['Museum'],relations=[dict(relation_id='r',traversal='forward')])])
    p=parsed('Car near Museum','near',9)
    assert len(select_direct_relations(p,obj)['relations'])==1
    p=parsed('Car in parking lot near Museum','near',24)
    assert not select_direct_relations(p,obj)['relations']
    p=parsed('Car behind Museum','behind',11)
    assert not select_direct_relations(p,obj)['relations']
    p=parsed('Car in Museum\'s parking lot','in',7)
    assert not select_direct_relations(p,obj)['relations']
    p['target_GT_used']=True
    with pytest.raises(ValueError): select_direct_relations(p,obj)
