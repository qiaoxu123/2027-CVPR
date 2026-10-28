import pytest
import torch
from experiments.a2_waldo_landmark_belief.direct_risk_head import DirectRiskHead,head_loss,outcome_classes,apply_direct_risk


def test_class_prior_correction_uses_saved_training_weights_only():
    model=DirectRiskHead(torch.zeros(15),torch.ones(15),torch.tensor([1.,2.,4.]))
    with torch.no_grad():model.linear.weight.zero_();model.linear.bias.zero_()
    logits=model(torch.zeros(2,15))
    assert torch.equal(logits,torch.zeros(2,3))
    assert torch.allclose(model.corrected_logits(logits),-torch.tensor([1.,2.,4.]).log().expand(2,-1))
    with pytest.raises(TypeError):model(torch.zeros(2,15),target_xy=torch.zeros(2))


def test_private_outcome_loss_has_finite_trainable_gradients_in_both_stages():
    torch.manual_seed(0);model=DirectRiskHead(torch.zeros(15),torch.ones(15),torch.ones(3))
    classes=outcome_classes(torch.tensor([0,1,-1,1]))
    assert classes.tolist()==[0,1,2,1]
    for refine in (False,True):
        model.zero_grad();loss,parts=head_loss(model(torch.randn(4,15)),classes,model.class_weights,refine=refine)
        loss.backward()
        assert torch.isfinite(loss) and model.linear.weight.grad.norm()>0 and torch.isfinite(model.linear.weight.grad).all()
    with pytest.raises(ValueError):outcome_classes(torch.tensor([2]))


def test_risk_gates_must_both_pass_and_no_evidence_returns_exact_A0():
    base=torch.tensor([.1,0.]);q=torch.tensor([0.,1.]);safe=torch.tensor([-10.,10.,-10.]);unsafe=torch.tensor([-10.,10.,10.])
    params=dict(enabled=True,minimum_gain=.5,maximum_harm=.1,temperature=1.)
    mixed,info=apply_direct_risk(base,q,safe,True,params)
    assert mixed.argmax()==1 and info['gate']==.25
    for logits,available,parameters in ((unsafe,True,params),(safe,False,params),(safe,True,dict(params,enabled=False))):
        unchanged,info=apply_direct_risk(base,q,logits,available,parameters)
        assert unchanged is base and info['gate']==0
