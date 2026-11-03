"""TRAIN-only positional weak supervision; no new inference inputs or rules."""
import torch
from torch.nn import functional as F
from .visual_chain_position import visual_chain_loss
from .local_chain_raster import rasterized_goal_logprob


def preservation_kl(prior_logits,updated_logits,world,target):
    prior=prior_logits.detach().float().log_softmax(-1)
    ids=prior.argmax(-1)
    position=world[torch.arange(len(world),device=world.device),ids]
    correct=(position-target[:,:2]).norm(dim=-1)<=20
    if not correct.any():return updated_logits.sum()*0,0
    per_row=F.kl_div(updated_logits.float().log_softmax(-1),prior.exp(),reduction='none').sum(-1)
    return per_row[correct].mean(),int(correct.sum())


def precision_preserve_loss(model,joint,visual,inputs,candidate_world,target_world,categories,phrases):
    terms=visual_chain_loss(model,joint,visual,inputs,candidate_world,target_world,categories,phrases)
    cells=inputs['prior_logits'].shape[-1];valid=inputs['local_valid']
    distance=(candidate_world[:,cells:].float()-target_world[:,None,:2]).norm(dim=-1)
    car=torch.tensor([c=='Car' for c in categories],device=visual.device)
    nearest=distance.masked_fill(~valid,float('inf')).amin(-1)
    # Only use fine Car point supervision when this observation supports it.
    fine=(nearest<=5)&(inputs['quality'][:,0,3]*100<=8)
    eligible=(valid&(distance<=20)).any(-1)&(~car|fine)
    sigma=torch.where(car,3.,10.)[:,None]
    positive=valid&(distance<=torch.where(car,10.,20.)[:,None])
    weights=torch.exp(-.5*(distance/sigma).square())*positive
    weights=weights/weights.sum(-1,keepdim=True).clamp_min(1e-12)
    if eligible.any():
        lp=visual.masked_fill(~valid,float('-inf'))[eligible].log_softmax(-1).masked_fill(~valid[eligible],0)
        local=-(weights[eligible]*lp).sum(-1).mean()
    else:local=visual.sum()*0
    updated=rasterized_goal_logprob(joint,inputs['local_uv'],valid,global_cells=cells)
    preserve,preserve_rows=preservation_kl(inputs['prior_logits'],updated,candidate_world[:,:cells],target_world)
    terms['total']=terms['total']-.5*terms['visual']+.5*local+.1*preserve
    terms.update(visual=local,preservation=preserve,preservation_rows=preserve_rows,local_supervised=int(eligible.sum()),fine_car_rows=int((car&fine&eligible).sum()))
    return terms
