import torch
from experiments.a2_waldo_landmark_belief.relation_v3_fusion import candidate_kernels,proposal,spatial_utility_labels


def test_stable_world_kernels_match_double_precision_reference_without_cdist_MM():
    torch.manual_seed(3);xy=torch.tensor([[[1234.2,1783.4],[1240.7,1794.1]]]);grid=torch.randn(1,784,2)*15+torch.tensor([1234.,1783.])
    s=dict(candidate_xy=xy,grid_world=grid,candidate_support=torch.ones(1,2,784,dtype=torch.bool))
    actual=candidate_kernels(s)
    reference=(-((xy.double()[:,:,None]-grid.double()[:,None]).square().sum(-1))/32).softmax(-1)
    assert torch.allclose(actual.double(),reference,atol=1e-7,rtol=1e-5)


def test_empty_support_has_no_nan_and_no_phantom_proposal():
    s=dict(candidate_xy=torch.zeros(1,2,2),grid_world=torch.zeros(1,3,2),candidate_support=torch.zeros(1,2,3,dtype=torch.bool))
    q,available=proposal(s,torch.zeros(1,2),torch.zeros(1,2),torch.ones(1,2,dtype=torch.bool))
    assert not available.any() and not q.any() and torch.isfinite(q).all()


def test_spatial_labels_describe_localization_not_unknown_object_identity():
    s=dict(candidate_xy=torch.tensor([[[0.,0.],[100.,0.]]]),grid_world=torch.tensor([[[0.,0.],[100.,0.]]]),candidate_support=torch.ones(1,2,2,dtype=torch.bool))
    utility=spatial_utility_labels(s,torch.tensor([[0.,0.]]))
    assert utility.tolist()==[[True,False]]
