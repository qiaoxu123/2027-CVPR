"""Weak TRAIN annotation-side instance cores; never call from model inference."""
import numpy as np


def instance_core_labels(candidate_world,valid,object_xy,object_ids,target_id,radius=2.,margin=.75):
    points=np.asarray(candidate_world,float);objects=np.asarray(object_xy,float);valid=np.asarray(valid);ids=np.asarray(object_ids)
    if points.ndim!=2 or points.shape[1]!=2 or objects.shape!=(len(ids),2) or valid.shape!=(len(points),) or valid.dtype!=bool:
        raise ValueError('world XY, registered object XY/IDs and bool validity required')
    if not np.isfinite(points).all() or not np.isfinite(objects).all() or radius<=0 or margin<0:raise ValueError('finite coordinates and valid uncertainty bounds required')
    if not len(ids) or len(set(ids.tolist()))!=len(ids) or target_id not in ids:raise ValueError('unique TRAIN IDs including target required')
    distance=np.linalg.norm(points[:,None]-objects[None],axis=-1)
    order=np.argsort(distance,axis=-1);nearest=order[:,0];first=distance[np.arange(len(points)),nearest]
    second=distance[np.arange(len(points)),order[:,1]] if len(ids)>1 else np.full(len(points),np.inf)
    known=valid&(first<=radius)&((second-first)>=margin)
    labels=np.full(len(points),-1,np.int8)
    labels[known]=(ids[nearest[known]]==target_id).astype(np.int8)
    return labels


def conservative_car_background(core_labels,candidate_world,valid,target_contour,buffer_m=2.):
    """Only far outside an expanded TRAIN contour is weak background; edges unknown."""
    from shapely.geometry import Polygon,Point
    labels=np.asarray(core_labels,np.int8).copy();points=np.asarray(candidate_world,float);valid=np.asarray(valid)
    if labels.shape!=(len(points),) or points.shape!=(len(labels),2) or valid.shape!=labels.shape or valid.dtype!=bool:
        raise ValueError('aligned core labels and public world positions required')
    contour=np.asarray(target_contour,float)
    if contour.ndim!=2 or contour.shape[0]<3 or contour.shape[1]!=2 or not np.isfinite(contour).all():return np.full_like(labels,-1)
    polygon=Polygon(contour)
    if polygon.is_empty or not polygon.is_valid or polygon.area<=0:return np.full_like(labels,-1)
    if not np.isfinite(buffer_m) or buffer_m<0:raise ValueError('finite nonnegative uncertainty buffer required')
    expanded=polygon.buffer(buffer_m);inside=np.array([expanded.covers(Point(xy)) for xy in points])
    conflict=(labels==1)&~inside
    labels[valid&~inside&~conflict]=0
    labels[~valid|conflict]=-1
    return labels


def instance_annotation_counts(core,final):
    """Keep registered-instance identities separate from added weak background."""
    core=np.asarray(core);final=np.asarray(final)
    if core.shape!=final.shape or core.ndim!=1:raise ValueError('matching one-frame label arrays required')
    return dict(frames_with_target_core=int((core==1).any()),
        frames_with_target_and_other_car_core=int((core==1).any() and (core==0).any()),
        retained_target_core_frames=int((final==1).any()),
        eligible_loss_frames=int((final==1).any() and (final==0).any()),
        background_negative_points=int(((final==0)&(core==-1)).sum()))
