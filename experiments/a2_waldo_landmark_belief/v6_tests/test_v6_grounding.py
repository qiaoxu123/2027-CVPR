import torch
from torch import nn
from experiments.a2_waldo_landmark_belief.v6_grounding import RelationalGrounder, candidate_proposal, selective_belief_update, weak_spatial_ranking_loss, oracle_diagnostic

class Evidence(nn.Module):
    def __init__(self):
        super().__init__()
        self.param = nn.Parameter(torch.ones(()))
    def forward(self, **network):
        b,n=network["box_valid"].shape
        return {"scores":torch.tensor([2.,1.]).expand(b,n), "candidate_match_logits":torch.zeros(b,n), "reliable":network["box_valid"]}

def sample():
    net={"box_rgb":torch.zeros(1,2,1024),"box_valid":torch.tensor([[True,True]]),"box_kind":torch.zeros(1,2,dtype=torch.long),"pair_geometry":torch.zeros(1,2,1,14),"pair_valid":torch.ones(1,2,1,dtype=torch.bool)}
    spatial={"base_logits":torch.tensor([[3.,0.,0.,0.]]),"grid_world":torch.tensor([[[0.,0.],[20.,0.],[0.,20.],[20.,20.]]]),"candidate_xy":torch.tensor([[[0.,0.],[20.,20.]]]),"candidate_support":torch.tensor([[[1,0,0,0],[0,0,0,1]]],dtype=torch.bool)}
    labels={"target_xy":torch.tensor([[20.,20.]]),"localization_utility":torch.tensor([[False,True]]),"utility_valid":torch.tensor([[True,True]]),"sample_weight":torch.tensor([1.])}
    return net,spatial,labels

def test_zero_initialized_residual_and_frozen_evidence():
    net,sp,_=sample()
    model=RelationalGrounder(Evidence())
    pred=model(net,sp)
    assert torch.allclose(pred["logits"],torch.tensor([[2.,1.]]))
    assert not any(p.requires_grad for p in model.evidence.parameters())

def test_default_and_no_candidate_exact_a0_fallback():
    net,sp,_=sample()
    model=RelationalGrounder(Evidence())
    pred=model(net,sp)
    result,active=selective_belief_update(sp,pred)
    assert torch.equal(result,sp["base_logits"]) and not active.any()
    net["box_valid"][:]=False
    pred=model(net,sp)
    result,active=selective_belief_update(sp,pred,enabled=True)
    assert torch.equal(result,sp["base_logits"]) and not active.any()

def test_weak_labels_and_oracle_separation():
    net,sp,labels=sample()
    model=RelationalGrounder(Evidence())
    pred=model(net,sp)
    loss,stats=weak_spatial_ranking_loss(pred,labels)
    assert torch.isfinite(loss)
    loss.backward()
    assert model.output.weight.grad is not None
    oracle=oracle_diagnostic(sp,labels,pred)
    assert oracle["covered"]==1 and oracle["a0_wrong_and_oracle_usable"]==1

def test_unknown_candidate_excluded_not_negative():
    net,sp,labels=sample()
    model=RelationalGrounder(Evidence())
    pred=model(net,sp)
    labels["utility_valid"][:,0]=False
    loss,stats=weak_spatial_ranking_loss(pred,labels)
    assert torch.isfinite(loss) and stats["contrast_rows"]==0
