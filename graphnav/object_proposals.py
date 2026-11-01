"""GT-free observed-object filtering and selection, with shared validity rules."""
import numpy as np
import torch
from torchvision.ops import nms


def box_valid_fraction(boxes,valid_pixels):
    boxes=np.asarray(boxes);height,width=valid_pixels.shape;quality=[]
    for cx,cy,w,h in boxes:
        if not np.isfinite([cx,cy,w,h]).all() or w<=0 or h<=0:
            quality.append(0.);continue
        raw=np.array([cx-w/2,cy-h/2,cx+w/2,cy+h/2])
        # A padded/outside box does not acquire validity from clipping alone.
        clipped=np.clip(raw,0,1);retained=max(0,clipped[2]-clipped[0])*max(0,clipped[3]-clipped[1])/(w*h)
        x0,y0,x1,y1=np.round(clipped*np.array([width-1,height-1,width-1,height-1])).astype(int)
        patch=valid_pixels[y0:y1+1,x0:x1+1]
        quality.append(float(patch.mean())*min(1.,retained) if patch.size else 0.)
    return np.asarray(quality,np.float32)


def select_proposals(boxes,scores,quality,budget=15,geometry=None,spatial=False):
    """NMS then optionally combine frozen geometry and spatial diversity. No labels."""
    if budget<=0:raise ValueError('budget must be positive')
    valid=torch.isfinite(scores)&torch.isfinite(boxes).all(-1)&(quality>=.6)&(boxes[:,2:]>0).all(-1)
    ids=valid.nonzero().flatten()
    if not len(ids):return ids
    ranking=torch.logit(scores.float().clamp(1e-5,1-1e-5))
    if geometry is not None:
        relative=geometry.float()-geometry[valid].amax()
        ranking=ranking+relative.clamp(-8,0)*.5
    corners=torch.cat((boxes[:,:2]-boxes[:,2:]/2,boxes[:,:2]+boxes[:,2:]/2),-1)
    ids=ids[nms(corners[ids].float(),ranking[ids].float(),.5)]
    if not spatial:return ids[:budget]
    # Deterministic soft spatial suppression preserves score ranking without GT recall tuning.
    remaining=ranking[ids].clone();chosen=[]
    for _ in range(min(budget,len(ids))):
        j=remaining.argmax();chosen.append(ids[j])
        distance=(boxes[ids,:2]-boxes[ids[j],:2]).norm(dim=-1)
        remaining=remaining-1.5*torch.exp(-distance.square()/(2*.12**2))
        remaining[j]=-float('inf')
    return torch.stack(chosen)
