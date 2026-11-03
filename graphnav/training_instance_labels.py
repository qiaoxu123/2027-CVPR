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
