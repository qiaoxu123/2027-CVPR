import numpy as np
import torch
from graphnav.bidirectional import verified_hard_negatives
from graphnav.bidirectional_train import hard_negative_margin


def landmark(identity,x,kind='Building'):
    return dict(id=identity,object_type=kind,center_world=np.array([x,0.]),center_norm=[x/200+.5,.5],contour=[[x-2,-2],[x+2,-2],[x+2,2],[x-2,2]])


def test_alternative_valid_target_is_never_a_negative():
    anchor,target=landmark('a',0),landmark('t',80)
    entries=[anchor,target,landmark('valid',90),landmark('wrong',-75),landmark('type',-80,'Water'),landmark('far',-180)]
    assert verified_hard_negatives(anchor,target,entries,'east')==[entries[3]['center_norm']]


def test_margin_has_no_effect_without_verified_negatives_and_pushes_wrong_score_down():
    grid=torch.tensor([[[0.,0.],[1.,0.]]]);positive=torch.tensor([[1.,0.]])
    negative=torch.tensor([[[0.,0.]]]);scores=torch.tensor([[2.,0.]],requires_grad=True)
    assert hard_negative_margin(scores,grid,positive,negative,torch.tensor([[False]])).item()==0
    loss=hard_negative_margin(scores,grid,positive,negative,torch.tensor([[True]]));loss.backward()
    assert scores.grad.tolist()==[[1.,-1.]]
