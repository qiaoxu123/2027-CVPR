"""Offline GT supervision ONLY. Runtime perception/policy must not import this."""
import numpy as np
import cv2
import torch
from .data_protocol import split_target_groups
from .core import _center


class PrivateAssociationIndex:
    def __init__(self,catalog):
        from shapely.geometry import Polygon,Point
        self.maps={}
        for name,objects in catalog.objects.items():
            rows=[]
            for oid,obj in objects.items():
                kind={'Car':0,'Building':1}.get(obj['object_type'],-1)
                if kind<0:
                    continue
                poly=np.asarray(obj['contour'],np.float32)
                shape=Polygon(poly)
                if not shape.is_valid or shape.area<=1e-6:
                    continue
                split=split_target_groups([dict(map_name=name,object_id=oid)],seed=0)[0]
                extent=poly.max(0)-poly.min(0)
                circularity=4*np.pi*shape.area/max(shape.length**2,1e-6)
                coarse_car=kind==0 and len(poly)>=8 and extent.min()>=6 and extent.max()/extent.min()<=1.6 and circularity>=.7
                position=np.asarray(obj.get('position',[]),np.float64).reshape(-1)
                anchor_valid=bool(len(position)>=2 and np.isfinite(position[:2]).all())
                anchor=position[:2].astype(np.float32) if anchor_valid else _center(poly)
                anchor_in_region=bool(anchor_valid and shape.covers(Point(anchor)))
                rows.append(dict(oid=int(oid),kind=kind,shape=shape,center=_center(poly),
                                 diagonal=float(np.linalg.norm(extent)),split=split,coarse_car=coarse_car,
                                 anchor=anchor,anchor_valid=anchor_valid,anchor_in_region=anchor_in_region))
            self.maps[name]=(rows,np.asarray([r['shape'].bounds for r in rows]).reshape(-1,4))

    def associate(self,map_name,candidates,target_id,split):
        from shapely.geometry import Polygon
        rows,bounds=self.maps[map_name]
        target_kind=next((r['kind'] for r in rows if r['oid']==int(target_id)),None)
        car_rows=[r for r in rows if r['kind']==0]
        car_centers=np.asarray([r['anchor'] for r in car_rows]).reshape(-1,2)
        car_ids=[i for i,c in enumerate(candidates) if c['kind']==0]
        detected_centers=np.asarray([np.asarray(candidates[i]['poly']).mean(0) for i in car_ids]).reshape(-1,2)
        truth=np.zeros(max(len(candidates),1),np.float32)
        known=np.zeros_like(truth,bool)
        details=[]
        for i,candidate in enumerate(candidates):
            poly=np.asarray(candidate['poly'],np.float32)
            shape=Polygon(poly)
            lo,hi=poly.min(0),poly.max(0)
            overlap=(bounds[:,0]<=hi[0]) & (bounds[:,2]>=lo[0]) & (bounds[:,1]<=hi[1]) & (bounds[:,3]>=lo[1])
            choices=[]
            if shape.is_valid and shape.area>0:
                for j in np.flatnonzero(overlap):
                    other=rows[j]
                    if other['kind']!=candidate['kind']:
                        continue
                    intersection=shape.intersection(other['shape']).area
                    iou=float(intersection/max(shape.area+other['shape'].area-intersection,1e-9))
                    distance=float(np.linalg.norm(poly.mean(0)-other['center'])/max(other['diagonal'],1e-6))
                    choices.append((iou,distance,other))
            choices.sort(key=lambda c:(-c[0],c[1],c[2]['oid']))
            best=choices[0] if choices else None
            second=choices[1][0] if len(choices)>1 else 0.
            trusted=bool(best and best[0]>=.5 and best[1]<=.5 and best[0]-second>=.15)
            method='strict_polygon_iou'
            anchor_details={}
            if candidate['kind']==0 and len(car_rows)>0:
                distances=np.linalg.norm(car_centers-poly.mean(0),axis=-1)
                order=np.argsort(distances,kind='stable')
                anchor=car_rows[order[0]]
                if anchor['coarse_car']:
                    # CityRefer car circles are coarse click regions, not car
                    # segmentations. Use a symmetric unique observed-box/GT
                    # anchor association; adjacent/duplicate detections remain
                    # unknown. Thresholds are TRAIN-audited, never benchmark-fit.
                    closest=float(distances[order[0]])
                    ann_margin=float(distances[order[1]]-closest) if len(order)>1 else None
                    reverse=np.linalg.norm(detected_centers-anchor['anchor'],axis=-1)
                    rev_order=np.argsort(reverse,kind='stable')
                    det_margin=float(reverse[rev_order[1]]-reverse[rev_order[0]]) if len(rev_order)>1 else None
                    contains=cv2.pointPolygonTest(poly,tuple(map(float,anchor['anchor'])),False)>=0
                    inside_anchors=sum(r['anchor_valid'] and cv2.pointPolygonTest(poly,tuple(map(float,r['anchor'])),False)>=0 for r in car_rows)
                    trusted=bool(anchor['anchor_valid'] and anchor['anchor_in_region'] and inside_anchors==1 and
                                 closest<=1. and (ann_margin is None or ann_margin>=1.) and
                                 car_ids[rev_order[0]]==i and (det_margin is None or det_margin>=1.) and
                                 contains and candidate.get('confidence',0.)>=.5)
                    method='unique_coarse_car_anchor' if trusted else 'unresolved_coarse_car_anchor'
                    selected=next((choice for choice in choices if choice[2]['oid']==anchor['oid']),None)
                    best=selected if selected else (0.,closest/max(anchor['diagonal'],1e-6),anchor)
                    second=max((choice[0] for choice in choices if choice[2]['oid']!=anchor['oid']),default=0.)
                    anchor_details=dict(center_distance_m=closest,annotation_center_margin_m=ann_margin,
                                        detection_center_margin_m=det_margin,anchor_world_xy=anchor['anchor'].tolist(),
                                        anchor_in_annotation_region=anchor['anchor_in_region'],anchors_in_box=inside_anchors)
                elif best and best[2]['coarse_car']:
                    # A coarse region cannot bypass anchor checks just because
                    # a different, noncoarse annotation has a closer position.
                    trusted=False
                    method='unresolved_coarse_car_anchor'
            # Identity is first resolved against ALL same-category objects.
            # Only then mask cross-partition negative supervision.
            allowed=bool(trusted and best[2]['split']==split and candidate['kind']==target_kind)
            known[i]=allowed
            truth[i]=float(allowed and best[2]['oid']==int(target_id))
            details.append(dict(object_id=best[2]['oid'] if best else None,iou=best[0] if best else 0.,
                                second_iou=second,normalized_distance=best[1] if best else None,
                                trusted=trusted,supervision_allowed=allowed,
                                associated_split=best[2]['split'] if best else None,method=method,**anchor_details))
        if truth.sum():
            truth/=truth.sum()
        return truth,known,details


def label_view(record,catalog,index,pack,meta,observation,split):
    from graphnav.landmark_projection import world_to_frame
    target=np.asarray(record.target_position[:2],np.float32)
    truth,known,association=index.associate(record.map_name,meta['candidates'],record.object_id,split)
    refs=meta['references']
    refxy=np.asarray([_center(np.asarray(r['contour'],np.float32)) for r in refs],np.float32).reshape(-1,2)
    delta=target[None]-refxy
    distance=np.linalg.norm(delta,axis=-1)
    l=max(len(refs),1)
    angle=np.zeros((l,2),np.float32)
    logdist=np.zeros(l,np.float32)
    angle_valid=np.zeros(l,bool)
    angle[:len(refs)]=delta/np.maximum(distance[:,None],1e-6)
    logdist[:len(refs)]=np.log1p(distance)/np.log1p(40)
    angle_valid[:len(refs)]=distance>1
    xy=pack['spatial']['candidate_xy'].detach().cpu().numpy()
    limit=pack['network']['offset_limit'].detach().cpu().numpy()
    offset=target[None]-xy
    pixel=world_to_frame(target[None],observation['footprint_world'],observation['bgr'].shape)[0]
    visible=bool(np.isfinite(pixel).all() and (pixel>=0).all() and (pixel<=383).all())
    if visible:
        visible=bool(observation['bgr'][int(pixel[1]),int(pixel[0])].max()>0)
    offset_valid=(truth>0) & (np.abs(offset)<=limit).all(-1) & visible
    match=bool(offset_valid.any())
    match_valid=match or (len(meta['candidates'])>0 and bool(known.all()) and not bool((truth>0).any()))
    labels=dict(ranking=truth,rank_label_mask=known,angle=angle,angle_valid=angle_valid,distance=logdist,
                offset=offset,offset_valid=offset_valid,match=float(match),match_valid=match_valid,target_xy=target)
    target_object=catalog.target_object(record)
    private=dict(association=association,target_visible=visible,target_polygon=target_object['contour'],
                 target_kind={'Car':0,'Building':1}.get(target_object['object_type'],-1))
    return {k:torch.as_tensor(v) for k,v in labels.items()},private
