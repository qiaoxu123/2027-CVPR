"""Geometry of actual observed RGB/depth and candidate-local belief support."""
import math
import cv2
import numpy as np
import torch


def sample_surface(surface, valid, polygon):
    """Weak visible interior DSM statistic, never object-top/extent supervision."""
    result = dict(z=None, valid=False, coverage=0., visible_fraction=0., spread=None)
    p = np.asarray(polygon, np.float32)
    if p.ndim != 2 or p.shape[1] != 2 or len(p)<3 or not np.isfinite(p).all():
        return result
    if np.abs(p).max()>1e6:
        return result
    area = abs(cv2.contourArea(p))
    if area<4:
        return result
    mask = np.zeros(surface.shape, np.uint8)
    cv2.fillPoly(mask, [np.rint(p).astype(np.int32)], 1)
    result['visible_fraction'] = min(1., float(mask.sum())/area)
    mask = cv2.erode(mask, np.ones((3,3),np.uint8), borderType=cv2.BORDER_CONSTANT, borderValue=0)>0
    good = mask & np.asarray(valid,bool) & np.isfinite(surface)
    result['coverage'] = float(good.sum()/max(mask.sum(),1))
    if result['visible_fraction']<.8 or result['coverage']<.7 or good.sum()<4:
        return result
    low, high = np.quantile(surface[good],[.1,.9])
    result['spread'] = float(high-low)
    if high-low>8:
        return result
    result.update(z=float(np.median(surface[good])), valid=True)
    return result


def local_support(grid, polygons, observed):
    """Projected polygon +2m, with an observed nearest-cell fallback <=20m.

    No A0 peak filter or selected-candidate truncation. A coarse 28x28 grid can
    have no supported cell for a real detection; this remains explicit fallback.
    """
    grid = np.asarray(grid,np.float32)
    support = np.zeros((len(polygons),len(grid)),bool)
    for i, polygon in enumerate(polygons):
        p=np.asarray(polygon,np.float32)
        inside=np.array([cv2.pointPolygonTest(p,tuple(map(float,point)),True)>=-2 for point in grid])
        support[i] = inside & observed
        if not support[i].any() and np.any(observed):
            center=p.mean(0)
            distance=np.linalg.norm(grid-center,axis=-1)
            distance[~observed]=np.inf
            nearest=int(distance.argmin())
            if distance[nearest]<=20:
                support[i,nearest]=True
    return support


def gaussian_belief_mix(base, grid, xy, scores, support, eligible, alpha):
    """Mass-preserving local Gaussian update with bounded learned box offsets."""
    if not math.isfinite(float(alpha)) or not 0<=float(alpha)<=.5:
        raise ValueError('alpha must be finite in [0,.5]')
    if float(alpha)==0:
        return base
    if base.ndim!=1 or support.shape!=(len(xy),len(base)) or eligible.shape!=(len(xy),):
        raise ValueError('misaligned local fusion tensors')
    usable=eligible.bool() & support.bool().any(-1)
    if not usable.any():
        return base
    support=support[usable].bool()
    union=support.any(0)
    kernel_logits=-.5*(torch.cdist(xy[usable].float(),grid.float())/4).square()
    kernel_logits=kernel_logits.masked_fill(~support,-torch.inf)
    kernels=kernel_logits.softmax(-1)
    proposal=(scores[usable].float().softmax(0)[:,None]*kernels).sum(0)
    # Keep log scale, including the EXACT outside values. logsumexp avoids
    # losing support mass when A0 is very confident in an unrelated cell.
    log_mass=base[union].float().logsumexp(0)
    log_proposal=proposal[union].clamp_min(1e-38).log()
    mixed=torch.logaddexp(base[union].float()+math.log1p(-float(alpha)),
                         log_mass+math.log(float(alpha))+log_proposal)
    result=base.clone()
    result[union]=mixed.to(base.dtype)
    return result
