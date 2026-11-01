import torch
from graphnav.object_query_belief import ObjectQueryRanker,object_evidence_loss


def test_null_and_unknown_evidence_preserve_prior_and_zero_init():
    torch.manual_seed(0);m=ObjectQueryRanker();features=torch.randn(2,4,256);text=torch.randn(2,7,256);mask=torch.ones(2,7,dtype=torch.bool);geo=torch.randn(2,4,8);valid=torch.ones(2,4,dtype=torch.bool)
    q,n=m(features,text,mask,geo,valid);base=torch.randn(2,9);observed=torch.ones(2,9,dtype=torch.bool);observed[:,0]=False;kernel=torch.rand(2,9,4)
    updated,delta=m.update_belief(base,q,n,kernel,observed,valid);assert torch.equal(updated,base) and not delta.any()
    with torch.no_grad():m.fusion_gain.fill_(.3)
    updated,delta=m.update_belief(base,q,n,kernel,observed,valid);assert not delta[:,0].any() and torch.isfinite(updated).all()
    valid.zero_();q,n=m(features,text,mask,geo,valid);updated,delta=m.update_belief(base,q,n,kernel,observed,valid)
    assert not delta.any() and torch.isfinite(updated).all()


def test_end_to_end_loss_trains_rank_and_fusion_without_labels_in_forward():
    torch.manual_seed(7);m=ObjectQueryRanker();f=torch.randn(2,4,256);t=torch.randn(2,7,256);mask=torch.ones(2,7,dtype=torch.bool);v=torch.ones(2,4,dtype=torch.bool);g=torch.randn(2,4,8)
    q,n=m(f,t,mask,g,v);base=torch.randn(2,9);world=torch.rand(2,9,2)*100;query_world=torch.rand(2,4,2)*100;target=torch.zeros(2,3)
    updated,_=m.update_belief(base,q,n,torch.rand(2,9,4),torch.ones(2,9,dtype=torch.bool),v)
    loss=object_evidence_loss(q,n,query_world,target,v,updated,world,base);loss['total'].backward()
    assert torch.isfinite(loss['total']) and m.score[-1].weight.grad.abs().sum()>0 and m.fusion_gain.grad.abs()>0


def test_official_relation_mixed_precision_index_writeback():
    import pytest
    if not torch.cuda.is_available():pytest.skip('CUDA BF16 regression')
    torch.manual_seed(11);model=ObjectQueryRanker(relation=True,train_relation=True).cuda()
    features=torch.randn(1,3,256,device='cuda');text=torch.randn(1,7,256,device='cuda');mask=torch.ones(1,7,dtype=torch.bool,device='cuda');valid=torch.ones(1,3,dtype=torch.bool,device='cuda');geometry=torch.randn(1,3,8,device='cuda')
    with torch.autocast('cuda',dtype=torch.bfloat16):q,null=model(features,text,mask,geometry,valid)
    assert q.dtype==torch.float32 and torch.isfinite(q).all()
    (q.mean()+null.float().mean()).backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    assert model.relation.layers[0]['cross_attn'].in_proj_weight.grad.abs().sum()>0
