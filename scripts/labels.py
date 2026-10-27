"""Private offline labels and masked objectives; never imported by runtime policy."""
import cv2
import numpy as np
import torch
from torch.nn import functional as F


def polygon_iou(a,b):
    a=np.asarray(a,np.float32);b=np.asarray(b,np.float32)
    lo=np.minimum(a.min(0),b.min(0));hi=np.maximum(a.max(0),b.max(0));scale=120/max(float((hi-lo).max()),1e-6)
    masks=[]
    for poly in (a,b):
        m=np.zeros((128,128),np.uint8);cv2.fillPoly(m,[np.rint((poly-lo)*scale+2).astype(np.int32)],1);masks.append(m>0)
    return float((masks[0]&masks[1]).sum()/max((masks[0]|masks[1]).sum(),1))


def associate(polygons,kinds,objects,target_id):
    n=len(polygons);rank=np.zeros(n,np.float32);mask=np.zeros(n,bool);details=[]
    for i,(poly,kind) in enumerate(zip(polygons,kinds)):
        choices=[];center=np.asarray(poly).mean(0)
        for oid,otype,contour in objects:
            if kind!=otype:continue
            p=np.asarray(contour,np.float32);iou=polygon_iou(poly,p);diag=float(np.linalg.norm(p.max(0)-p.min(0)))
            dist=float(np.linalg.norm(center-(p.min(0)+p.max(0))/2))
            choices.append((iou,int(oid),dist/max(diag,1e-6)))
        choices.sort(reverse=True)
        best=choices[0] if choices else (0.,-1,1e6);second=choices[1][0] if len(choices)>1 else 0.
        trusted=best[0]>=.5 and best[2]<=.5 and best[0]-second>=.15
        mask[i]=trusted;rank[i]=float(trusted and best[1]==target_id)
        details.append(dict(object_id=best[1],iou=best[0],second_iou=second,normalized_distance=best[2],trusted=trusted))
    if rank.sum():rank/=rank.sum()
    return dict(ranking=rank,rank_label_mask=mask,association=details)


def ranking_loss(scores,target,mask):
    if not bool((target[mask]>0).any()):return scores.sum()*0
    selected=target[mask];selected=selected/selected.sum().clamp_min(1e-8)
    return -(selected*scores[mask].log_softmax(0)).sum()


def objective(out,labels,offset=False):
    rank=ranking_loss(out['scores'],labels['ranking'],labels['rank_label_mask'])
    av=labels['angle_valid'];hv=labels['height_valid']
    angle=((1-(out['direction']*labels['angle']).sum(-1))*av).sum()/av.sum().clamp_min(1)
    distance=(F.smooth_l1_loss(out['distance'],labels['distance'],reduction='none')*av).sum()/av.sum().clamp_min(1)
    height=(F.smooth_l1_loss(out['height'],labels['height'],reduction='none')*hv).sum()/hv.sum().clamp_min(1)
    match=F.binary_cross_entropy_with_logits(out['match_logit'],labels['match']) if bool(labels['match_valid']) else out['match_logit']*0
    ov=labels['offset_valid'];delta=F.smooth_l1_loss(out['offset']/8,labels['offset']/8,reduction='none').mean(-1)
    delta=(delta*ov).sum()/ov.sum().clamp_min(1)
    loss=rank+.2*angle+.1*distance+.1*height+.2*match+(.5*delta if offset else delta*0)
    return loss,dict(rank=rank,angle=angle,distance=distance,height=height,match=match,offset=delta)
