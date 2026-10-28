import torch
from experiments.a2_waldo_landmark_belief.risk_probe import public_risk_features,outcome_labels


def test_public_features_require_no_target_and_are_finite_without_evidence():
    base=torch.tensor([[2.,0.]]);q=torch.tensor([[0.,0.]]);grid=torch.tensor([[[0.,0.],[100.,0.]]])
    features=public_risk_features(base,q,grid,torch.tensor([.6]),torch.tensor([False]))
    assert features.shape==(1,15) and torch.isfinite(features).all()
    assert features[0,-1]==0


def test_direct_outcome_labels_distinguish_gain_harm_and_no_change():
    base=torch.tensor([[.1,0.],[.1,0.],[.1,0.]])
    q=torch.tensor([[0.,1.],[0.,1.],[1.,0.]])
    grid=torch.tensor([[[0.,0.],[100.,0.]]]).expand(3,-1,-1)
    target=torch.tensor([[100.,0.],[0.,0.],[0.,0.]])
    result=outcome_labels(base,q,grid,target)
    assert result['outcomes'][:,-1].tolist()==[1,-1,0]
    assert result['before_ne'].tolist()==[100.,0.,0.]
