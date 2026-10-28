import torch
import pytest
from experiments.a2_waldo_landmark_belief.direct_risk_train import selection_key,train_epoch,verify_history
from experiments.a2_waldo_landmark_belief.direct_risk_head import DirectRiskHead


def test_operating_point_benefit_precedes_AUC_when_selecting_checkpoint():
    p=dict(index=0,gain_quantile=.5,harm_quantile=.1,summary=dict(mean_r1_difference=.01,broken_groups=0,mean_ne_difference_m=-1.,supported_groups=40))
    safe=dict(epoch=2,dev=dict(selected_point=p,gain_harm_auc=.7,classification=1.))
    empty=dict(epoch=1,dev=dict(selected_point=None,gain_harm_auc=.99,classification=.1))
    assert selection_key(safe)>selection_key(empty)


def test_epoch_updates_all_batches_without_changing_frozen_training_statistics():
    torch.manual_seed(0);model=DirectRiskHead(torch.zeros(15),torch.ones(15),torch.ones(3))
    optimizer=torch.optim.AdamW(model.parameters(),lr=.01)
    x=torch.randn(7,15);y=torch.tensor([0,1,2,0,1,2,0]);before=model.linear.weight.detach().clone()
    stats=train_epoch(model,x,y,optimizer,epoch=1,batch_size=3)
    assert stats['views']==7 and stats['optimizer_updates']==3 and not torch.equal(before,model.linear.weight)
    assert torch.equal(model.mean,torch.zeros(15)) and torch.equal(model.std,torch.ones(15))


def test_history_rejects_partial_schedule_and_wrong_update_counts():
    with pytest.raises(ValueError):verify_history([])
