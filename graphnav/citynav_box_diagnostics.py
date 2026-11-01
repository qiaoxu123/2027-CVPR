"""Offline TRAIN annotation/Oracle contracts; never imported by deployed policies."""
from dataclasses import dataclass
from enum import Enum
import numpy as np
import torch
from torchvision.ops import nms,box_iou
from .landmark_projection import frame_to_world


class OracleMode(str,Enum):
    NORMAL='normal'
    VERIFIED='verified'
    PROJECTION_HYPOTHESIS='projection-hypothesis'


@dataclass(frozen=True)
class BoxAuthorization:
    bbox: tuple
    provenance: str
    actual_visibility_verified: bool


def validate_box(box,shape):
    if box is None or len(box)!=4:raise ValueError('Missing pixel bbox')
    x0,y0,x1,y1=map(float,box);h,w=shape[:2]
    if not np.isfinite([x0,y0,x1,y1]).all() or not (0<=x0<x1<=w and 0<=y0<y1<=h):
        raise ValueError('Invalid exclusive pixel bbox')
    return (x0,y0,x1,y1)


def authorize_box(mode,annotation,review=None,partition='train'):
    """A missing/negative human review never becomes a verified Oracle."""
    if partition!='train' or OracleMode(mode)==OracleMode.NORMAL:
        raise ValueError('Private boxes are forbidden in normal/non-TRAIN inference')
    review=review or {}
    if review.get('status') in ['rejected','not_visible'] or review.get('visibility')=='not_visible':return None
    if OracleMode(mode)==OracleMode.VERIFIED:
        if not (review.get('status') in ['confirmed','corrected'] and review.get('source')=='human'
                and review.get('reviewer','').strip() and review.get('human_attested') is True
                and review.get('visibility') in ['visible','partially_visible']):return None
        if review.get('image_sha')!=annotation['image_sha'] or review.get('sample_id')!=annotation['sample_id']:
            raise ValueError('Review/image identity mismatch')
        box=review.get('bbox') if review.get('status')=='corrected' else annotation['auto_bbox_xyxy']
        return BoxAuthorization(validate_box(box,annotation['image_shape']),'human_review',True)
    if not annotation['quality']['high_geometric_support']:return None
    return BoxAuthorization(validate_box(annotation['auto_bbox_xyxy'],annotation['image_shape']),
                            'UNVERIFIED_projection_hypothesis_NOT_GT',False)


def bbox_cxcywh(box,shape):
    x0,y0,x1,y1=validate_box(box,shape);h,w=shape[:2]
    return torch.tensor([(x0+x1)/(2*w),(y0+y1)/(2*h),(x1-x0)/w,(y1-y0)/h])


def box_world_centers(boxes,footprint,shape):
    h,w=shape[:2]
    pixel=boxes[:,:2].cpu().numpy()*np.array([w-1,h-1],np.float32)
    return torch.from_numpy(frame_to_world(pixel,np.asarray(footprint,np.float32),shape))


def public_selection_stages(boxes,native_scores,quality,prior):
    """No label/instance/target argument. Exact deployed selector plus budgets."""
    valid=torch.isfinite(native_scores)&torch.isfinite(boxes).all(-1)&(quality>=.6)&(boxes[:,2:]>0).all(-1)
    ids=valid.nonzero().flatten()
    ranking=torch.logit(native_scores.float().clamp(1e-5,1-1e-5))
    if len(ids):ranking=ranking+.5*(prior-prior[valid].max()).clamp(-8,0)
    corners=torch.cat([boxes[:,:2]-boxes[:,2:]/2,boxes[:,:2]+boxes[:,2:]/2],-1)
    kept=ids[nms(corners[ids].float(),ranking[ids].float(),.5)] if len(ids) else ids
    chosen=[];remaining=ranking[kept].clone()
    for _ in range(min(60,len(kept))):
        j=remaining.argmax();chosen.append(kept[j]);distance=(boxes[kept,:2]-boxes[kept[j],:2]).norm(dim=-1)
        remaining-=1.5*torch.exp(-distance.square()/(2*.12**2));remaining[j]=-float('inf')
    order=torch.stack(chosen) if chosen else kept
    return dict(raw900=torch.arange(len(boxes),device=boxes.device),valid900=ids,nms=kept,
                top15=order[:15],top30=order[:30],top60=order[:60])


def registered_iou(boxes,pixel_box,shape):
    h,w=shape[:2];gt=torch.tensor(pixel_box,device=boxes.device,dtype=torch.float32)[None]/torch.tensor([w,h,w,h],device=boxes.device)
    corners=torch.cat([boxes[:,:2]-boxes[:,2:]/2,boxes[:,:2]+boxes[:,2:]/2],-1).clamp(0,1)
    return box_iou(corners.float(),gt)[:,0]


def bounded_fusion_limit(base,grid,observed,target,gain,bound=2.):
    """Optimistic analytic range; not an injected image/box or observed Oracle."""
    distance=(grid-target[:2]).norm(dim=-1);good=distance<=20
    before=int(base.argmax());a0_correct=bool(good[before])
    amplitude=float(torch.as_tensor(gain).clamp_min(0).tanh())*bound
    if not good.any():return dict(a0_correct=a0_correct,no_legal_grid=True,bounded_impossible=not a0_correct,amplitude=amplitude)
    upper=(base+amplitude*observed)[good].max()
    lower=base[before]-amplitude*observed[before]
    return dict(a0_correct=a0_correct,no_legal_grid=False,bounded_impossible=bool((upper<lower) and not a0_correct),
                amplitude=amplitude,optimistic_good_minus_bad=float(upper-lower),
                target_support=bool((good&observed).any()),a0_peak_observed=bool(observed[before]))
