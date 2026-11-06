import torch
from scripts.oracle_bearing_geometry import rerank,replace_field
from graphnav.full_localization import GoalField


def test_forward_ray_and_angles():
    xy=torch.tensor([[10.,0.],[-10.,0.],[10.,5.],[40.,0.]])
    z=torch.tensor([0.,100.,3.,2.]);a=[[0.,0.]]
    for mode in ['B','C','D']:
        q=rerank(z*0,xy,a,[0.],mode)
        assert q[0]>q[1] and q[0]>q[2]
    assert rerank(z,xy,a,[0.],'C')[1]<-1000
    assert rerank(z,xy,a,[0.],'B').argmax()==3 # angular tie broken by prior


def test_no_gt_range_and_scale():
    xy=torch.tensor([[10.,0.],[20.,0.],[10.,1.]])
    z=torch.zeros(3)
    assert torch.allclose(rerank(z,xy,[[0.,0.]],[0.],'D'),rerank(z,xy*2,[[0.,0.]],[0.],'D'))
    assert torch.equal(rerank(z,xy,[],[],'B'),z)
    assert torch.equal(rerank(z,xy,[[0.,0.]],[3.14159265],'C'),z)


def test_context_alignment_and_multi_anchor():
    z=torch.tensor([[0.,1.,2.]]);grid=torch.tensor([[[0.,0.],[1.,0.],[2.,0.]]]);ctx=torch.arange(6.).reshape(1,3,2)
    f=GoalField(z,torch.tensor([[2,1]]),grid[:,[2,1]],ctx[:,[2,1]],dense_context=ctx)
    out=replace_field(f,torch.tensor([[10.,2.,1.]]),grid)
    assert out.candidate_ids.tolist()==[[0,1]]
    assert torch.equal(out.candidate_context,ctx[:,:2])
    q=rerank(z[0],grid[0],[[0.,-1.],[0.,1.]],[0.,0.],'D')
    assert torch.isfinite(q).all()
