"""Public polygon-pair hull support, a coarse proxy rather than instance truth.

Each homonymous polygon pair is retained as an alternative. Full polygons,
including road surfaces, are never silently converted into centerlines.
"""
import numpy as np
import shapely
from shapely.geometry import Polygon


def between_distance(world_xy, endpoints, objects):
    xy=np.asarray(world_xy,dtype=float)
    if xy.ndim!=2 or xy.shape[1]!=2 or not np.isfinite(xy).all():
        raise ValueError('finite world XY hypotheses required')
    if len(endpoints)!=2:raise ValueError('exactly two scoped endpoints required')
    alternatives=[]
    for left in endpoints[0]['public_ids']:
        for right in endpoints[1]['public_ids']:
            if str(left)==str(right):continue
            polygons=[]
            for oid in (left,right):
                obj=objects[str(oid)];contour=np.asarray(obj['contour'],float)
                if contour.ndim!=2 or contour.shape[1]!=2 or len(contour)<3 or not np.isfinite(contour).all():break
                poly=Polygon(contour)
                if not poly.is_valid or poly.area<=0:break
                polygons.append(poly)
            if len(polygons)==2:
                # Hull is only a permissive corridor hypothesis, not a claim
                # that all its points are the target or even navigable.
                alternatives.append(shapely.union_all(polygons).convex_hull)
    if not alternatives:return np.zeros(len(xy),np.float32),False
    distances=[shapely.distance(shapely.points(xy),p) for p in alternatives]
    return np.min(distances,axis=0).astype(np.float32),True
