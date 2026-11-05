"""TRAIN-only privileged crop plans and existing PGVB world-grid refinement."""
import math
import numpy as np
import shapely
import torch
from .privileged_landmark_rgb import whole_landmark_view
from .landmark_region_supervision import named_union
from .memory import ObservedRGBMemory
from .world_likelihood_evidence import world_evidence_logits


def tile_plan(contours,scales=(128.,64.),context_m=32.):
    full=whole_landmark_view(contours);fp=np.asarray(full['footprint_world']);lo=fp.min(0);hi=fp.max(0);region=named_union(contours).buffer(context_m);tiles=[]
    for scale in scales:
        n=max(1,math.ceil(full['side_m']/scale))
        if n==1:continue
        side=full['side_m']/n
        for y in range(n):
            for x in range(n):
                center=lo+np.array([x+.5,y+.5])*side
                box=shapely.box(*(center-side/2),*(center+side/2))
                if not box.intersects(region):continue
                tiles.append(dict(footprint_world=ObservedRGBMemory.view_corners(center,0.,side/2),side_m=side,scale_m=scale))
    return dict(whole=full,tiles=tiles)


def refine_cached(model,base,dense,observed,*,joint=False):
    """Same bounded PGVB formula as forward_field; no new spatial rules.

    Multiple source views are averaged by existing world_evidence_logits,
    not counted as independent likelihood observations.
    """
    if observed is None:return base,torch.zeros_like(base,dtype=torch.bool)
    scores=model.pgvb(**{k:v for k,v in observed.items() if k!='local_uv'}).float();valid=observed['valid'];uv=observed['local_uv']
    center=scores-(scores*valid).sum(-1,keepdim=True)/valid.sum(-1,keepdim=True).clamp_min(1)
    zeros=torch.zeros_like(base);neutral=zeros.log_softmax(-1)
    evidence=world_evidence_logits(zeros,uv,center,valid)-neutral
    covered=(world_evidence_logits(zeros,uv,torch.ones_like(center),valid)-neutral)>.5
    update=evidence/5
    if joint:
        features=torch.cat((dense.float(),evidence[...,None],covered.float()[...,None]),-1)
        update=update+model.joint_logits(features).squeeze(-1)
    result=base+.5*torch.tanh(update)*covered
    return result,covered


def refine_views(model,base,dense,packs,*,joint=False,control='real'):
    """PGVB once per registered view, then average evidence on shared world grid."""
    device=base.device;present=[(i,x) for i,x in enumerate(packs) if x is not None]
    if not present:return base,torch.zeros_like(base,dtype=torch.bool)
    kwargs={k:torch.cat([x[k] for _,x in present]).to(device).float() for k in ('image_features','name_text','type_text','polygon_tokens')}
    valid=torch.cat([x['valid'] for _,x in present]).to(device).bool()
    if control=='zero':kwargs['image_features']=torch.zeros_like(kwargs['image_features'])
    elif control=='shuffle':
        # Keep all geometry, valid pixels and scale slots; break RGB/world correspondence.
        image=kwargs['image_features'].clone()
        for i in range(len(image)):
            ids=valid[i].nonzero().flatten()
            if len(ids)>1:image[i,ids]=image[i,ids].roll(max(1,len(ids)//2),0)
        kwargs['image_features']=image
    elif control!='real':raise ValueError('Unknown image control')
    scores=model.pgvb(**kwargs,valid=valid).float();scores=scores-(scores*valid).sum(-1,keepdim=True)/valid.sum(-1,keepdim=True).clamp_min(1)
    lengths=[x['valid'].numel() if x is not None else 0 for x in packs];maxp=max(lengths)
    uv=torch.zeros(len(packs),maxp,2,device=device);values=torch.zeros(len(packs),maxp,device=device);mask=torch.zeros(len(packs),maxp,dtype=torch.bool,device=device);offset=0
    for i,x in present:
        views=len(x['valid']);n=lengths[i];uv[i,:n]=x['local_uv'].to(device).flatten(0,1);values[i,:n]=scores[offset:offset+views].flatten();mask[i,:n]=valid[offset:offset+views].flatten();offset+=views
    zeros=torch.zeros_like(base);neutral=zeros.log_softmax(-1);evidence=world_evidence_logits(zeros,uv,values,mask)-neutral;covered=(world_evidence_logits(zeros,uv,torch.ones_like(values),mask)-neutral)>.5
    update=evidence/5
    if joint:update=update+model.joint_logits(torch.cat((dense.float(),evidence[...,None],covered.float()[...,None]),-1)).squeeze(-1)
    return base+.5*torch.tanh(update)*covered,covered
