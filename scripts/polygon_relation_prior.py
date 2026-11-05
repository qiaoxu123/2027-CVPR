"""Offline explicit metric/topological priors. No geographic-axis assumption."""
import numpy as np
import shapely

WORDS=('inside','outside','near')


def relation_prior(world,polygon,word,radius_m=30.):
    """Return unnormalized soft logits and a diagnostic region mask.

    Near=within 30m of the complete Polygon (including its interior).
    Inside uses signed boundary distance; outside its inverse. This is a
    linguistic geometric hypothesis, NOT a target instance or visual mask.
    """
    if word not in WORDS:
        raise ValueError('Direction requires an independently verified georeference')
    points=shapely.points(np.asarray(world))
    inside=shapely.covers(polygon,points)
    boundary=shapely.distance(points,polygon.boundary)
    signed=np.where(inside,boundary,-boundary)
    distance=shapely.distance(points,polygon)
    if word=='inside':return -np.logaddexp(0.,-signed/5.),inside
    if word=='outside':return -np.logaddexp(0.,signed/5.),~inside
    return -.5*(distance/radius_m)**2,distance<=radius_m
