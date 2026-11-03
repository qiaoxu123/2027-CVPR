import torch
from test_local_chain_position import inputs
from graphnav.visual_chain_position import VisualChainPositionHead,visual_chain_loss


def test_visual_score_cannot_use_geometry_or_prior_shortcuts():
    model=VisualChainPositionHead(dimension=12,hidden=8);x=inputs()
    _,a=model(**x,return_visual=True)
    x['public_geometry']*=100;x['prior_logits']*=10;x['quality']*=2
    _,b=model(**x,return_visual=True)
    torch.testing.assert_close(a,b)


def test_grid_and_visual_losses_backpropagate_without_pixel_gold():
    model=VisualChainPositionHead(dimension=12,hidden=8);x=inputs()
    joint,visual=model(**x,return_visual=True)
    axis=(torch.arange(4)+.5)/4;y,z=torch.meshgrid(axis,axis,indexing='ij');world=torch.stack((z.flatten(),y.flatten()),-1)[None].expand(2,-1,-1)*40
    local=x['local_uv']*40;coords=torch.cat((world,local),1)
    terms=visual_chain_loss(model,joint,visual,x,coords,local[:,0],['Car','Car'],['white car','black car'])
    terms['total'].backward()
    assert model.visual_score[0].weight.grad.norm()>0
    assert model.image.weight.grad.norm()>0
    assert torch.isfinite(terms['total'])
