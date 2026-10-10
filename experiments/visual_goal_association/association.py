"""Public observed-instance to unchanged grid association; no target labels."""
import numpy as np
import shapely
from shapely.geometry import Polygon

METHODS=('centroid','polygon_overlap','distance_soft','instance')


def grid_geometry(bounds, grid_xy):
    bounds=np.asarray(bounds,float); xy=np.asarray(grid_xy,float)
    if bounds.shape!=(4,) or xy.shape!=(784,2) or np.any(bounds[2:]<=bounds[:2]):
        raise ValueError('unchanged 28x28 grid required')
    points=bounds[:2]+xy*(bounds[2:]-bounds[:2]); size=(bounds[2:]-bounds[:2])/28
    cells=shapely.box(points[:,0]-size[0]/2,points[:,1]-size[1]/2,
                      points[:,0]+size[0]/2,points[:,1]+size[1]/2)
    return points,cells,size


def associate(contour, footprint, bounds, grid_xy, method, legacy_grid=None):
    """Clip evidence to visited FOV. No image implies no association.

    Cell support denotes cells intersecting the actual camera footprint; does
    not assert every pixel in a cell was seen or semantic segmentation truth.
    Soft support is finite (two cell diagonals), confined to that footprint.
    Instance retains footprint overlap and uses max, rather than mean, fusion.
    """
    if method not in METHODS:raise ValueError('unknown association')
    poly=Polygon(np.asarray(contour,float)); view=Polygon(np.asarray(footprint,float))
    if not poly.is_valid or not view.is_valid or poly.area<=0 or view.area<=0:
        raise ValueError('valid finite public polygons required')
    observed=poly.intersection(view)
    if observed.is_empty:return np.zeros(784,float)
    points,cells,size=grid_geometry(bounds,grid_xy)
    in_view=shapely.area(shapely.intersection(cells,view))>1e-8
    if method=='centroid':
        center=np.asarray(contour,float).mean(0)
        index=int(np.square(points-center).sum(1).argmin()) if legacy_grid is None else int(legacy_grid)
        w=np.zeros(784,float)
        if in_view[index]:w[index]=1
        return w
    if method in ('polygon_overlap','instance'):
        w=shapely.area(shapely.intersection(cells,observed))/(size.prod())
        # Tiny cars still carry an instance; normalization does not dilute them
        # merely because a cell is substantially larger than the polygon.
        return np.where(in_view,w/max(w.max(),1e-12),0)
    sigma=float(np.linalg.norm(size)/2)
    distances=shapely.distance(shapely.points(points),observed)
    w=np.exp(-.5*(distances/sigma)**2)
    return np.where(in_view & (distances<=2*sigma),w,0)


def calibrated_scores(scores,confidence,quality,config):
    scores=np.asarray(scores,float); confidence=np.asarray(confidence,float)
    if not len(scores):return scores
    if config['score_mode']=='raw': values=scores
    elif config['score_mode']=='standardized':
        values=np.clip((scores-scores.mean())/max(float(scores.std()),.05),-3,3)
    else:raise ValueError('invalid score calibration')
    reliability=confidence if config['confidence']=='learned' else np.asarray(quality,float)
    return values*np.clip(reliability,0,1)


def fuse(logits,weights,values,*,gain,method):
    """Unknown cell logits exactly unchanged; polygon-overlap weights public."""
    logits=np.asarray(logits,float); weights=np.asarray(weights,float); values=np.asarray(values,float)
    if weights.shape!=(len(values),len(logits)) or not np.isfinite(weights).all() or np.any(weights<0):
        raise ValueError('finite nonnegative observed association required')
    known=weights.sum(0)>0
    residual=np.zeros_like(logits)
    if known.any() and len(values):
        if method=='instance':
            # Preserve strongest instance after aggregating its actual views;
            # averaging different objects sharing a cell destroys identity.
            v=values-values.mean()
            field=np.max(np.where(weights>0,weights*v[:,None],-np.inf),axis=0)
            residual[known]=field[known]
        else:
            field=(weights*values[:,None]).sum(0)/weights.sum(0).clip(1e-12)
            residual[known]=field[known]-field[known].mean()
    residual*=float(gain)
    updated=logits+residual
    assert np.array_equal(updated[~known],logits[~known])
    return updated,known,residual


def identity_oracle(logits,weights,target_mask):
    """Offline identity ceiling. Does not know target coordinates or best cell.

    An identity oracle boosts *all* associated cells of the observed target
    above the static maximum; static prior resolves within-instance ambiguity.
    This is not a legal visual score or a deployable inference policy.
    """
    logits=np.asarray(logits,float);weights=np.asarray(weights,float)
    support=(weights[np.asarray(target_mask,bool)]>0).any(0) if np.any(target_mask) else np.zeros(len(logits),bool)
    residual=np.zeros_like(logits)
    if support.any():residual[support]=float(logits.max()-logits.min()+1)
    return logits+residual,support,residual
