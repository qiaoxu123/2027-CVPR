"""Tests for new offline diagnostics; production implementation is unchanged."""
from types import SimpleNamespace
import inspect
import torch
from scripts.diagnose_v3_observation_fusion import candidate_changes,score

def test_counterfactuals_cannot_update_unobserved_cells():
    g=torch.linspace(-2,2,384*384).reshape(1,384,384)
    out=SimpleNamespace(geometry_global=g,visual_global=torch.ones_like(g)*12,
                        observed_mask=torch.zeros_like(g,dtype=torch.bool))
    before=g.clone()
    alternatives,meta=candidate_changes(out,1.,torch.tensor([0.,0.,500.,500.]))
    assert all(torch.equal(x,g[0]) for x in alternatives.values())
    assert torch.equal(g,before) and meta['visual_public_gap']==0

def test_original_scale_and_confidence_rule_use_no_target_input():
    assert 'target' not in inspect.signature(candidate_changes).parameters
    g=torch.zeros(1,384,384);v=torch.full_like(g,-10.);mask=torch.ones_like(g,dtype=torch.bool)
    out=SimpleNamespace(geometry_global=g,visual_global=v,observed_mask=mask)
    alternatives,_=candidate_changes(out,.8,torch.tensor([0.,0.,500.,500.]))
    assert torch.allclose(alternatives['scale1'],g[0]+.8*torch.tanh(v[0]/5))
    assert torch.equal(alternatives['margin_confidence'],g[0])
    assert torch.equal(alternatives['centered'],g[0])
