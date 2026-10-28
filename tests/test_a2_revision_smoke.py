import torch
from experiments.a2_waldo_landmark_belief.revision_smoke import meaningful_view


def test_cuda_gate_requires_full_competition_with_reachable_positive():
    x=dict(link_valid=torch.tensor([True]))
    y=dict(ranking=torch.tensor([1.,0.]),rank_label_mask=torch.tensor([True,True]),candidate_trust=torch.tensor([True,False]))
    row=dict(public=dict(network=x),labels=y)
    assert meaningful_view(row)
    y['rank_label_mask'][1]=False;assert not meaningful_view(row)
    y['rank_label_mask'][1]=True;y['candidate_trust'].zero_();assert not meaningful_view(row)
