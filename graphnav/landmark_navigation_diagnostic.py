"""Public-polygon navigation metrics and explicitly diagnostic Goal intervention."""
import numpy as np
import shapely
from shapely.geometry import Point, Polygon, box
from shapely.ops import nearest_points


def public_region(polygons, bounds=None):
    regions=[]
    for contour in polygons:
        poly=Polygon(contour)
        if not poly.is_valid or poly.area<=0:
            raise ValueError('Invalid public polygon; do not silently repair')
        regions.append(poly)
    if not regions:
        raise ValueError('No public polygon')
    region=shapely.union_all(regions)
    if bounds is not None:
        region=region.intersection(box(*bounds))
    if region.is_empty or region.area<=0:
        raise ValueError('Public landmark has no positive-area map intersection')
    return region


def oracle_goal_xy(region, start_xy):
    # A closest legal map point, never the centroid of an elongated road.
    return np.asarray(nearest_points(Point(start_xy),region)[1].coords[0],dtype=float)


def polygon_trace_metrics(poses, region, termination, radius=20.):
    path=np.asarray(poses,float)[:,:2]
    distance=shapely.distance(shapely.points(path),region)
    return dict(SR=float(distance[-1]<=radius),OSR=float(distance.min()<=radius),
        NE=float(distance[-1]),min_polygon_distance_m=float(distance.min()),
        initial_polygon_distance_m=float(distance[0]),steps=len(path)-1,
        path_length_m=float(np.linalg.norm(np.diff(path,axis=0),axis=-1).sum()),
        policy_stop=float(termination=='policy_stop'),
        false_stop=float(termination=='policy_stop' and distance[-1]>radius),
        first_entry_step=next((int(i) for i,x in enumerate(distance) if x<=radius),None))


def oracle_goal_field(field, grid_uv, bounds, goal_xy, *, diagnostic=False):
    """Replace goal only; preserve pretrained candidate contexts and head.

    Continuous candidate zero is the requested public-polygon point, with
    context from its nearest original grid site. This interface mismatch is
    explicitly logged; no new learned layer or fake visual feature is added.
    """
    if not diagnostic:
        raise PermissionError('Oracle intervention is diagnostic only')
    import torch
    from .full_localization import GoalField
    goal=torch.as_tensor(goal_xy,device=grid_uv.device,dtype=torch.float32).view(1,2)
    world=bounds[:,None,:2]+grid_uv.float()*(bounds[:,None,2:]-bounds[:,None,:2])
    distance=(world-goal[:,None]).square().sum(-1)
    logits=-distance/(2*20.**2)
    ids=distance.argsort(dim=-1,stable=True)[:,:field.candidate_ids.shape[1]]
    xy=grid_uv.gather(1,ids[...,None].expand(-1,-1,2)).clone().float()
    xy[:,0]=(goal-bounds[:,:2])/(bounds[:,2:]-bounds[:,:2])
    if (xy<0).any() or (xy>1).any():
        raise ValueError('Oracle goal outside legal map bounds')
    dense=field.dense_context
    if dense is None:
        raise ValueError('Original dense context required, no fabricated query features')
    context=dense.gather(1,ids[...,None].expand(-1,-1,dense.shape[-1]))
    return GoalField(logits,ids,xy,context,visual_coverage=field.visual_coverage,dense_context=dense)
