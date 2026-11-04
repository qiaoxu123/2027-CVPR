"""Public named Polygon intersections, never inferred target instance labels."""
import cv2
import numpy as np
import shapely
from .landmark_alias import canonical_name
from .landmark_projection import world_to_frame
from .landmark_region_supervision import named_union,weak_region_labels
from .native_dense_regions import SPARSE_IDS
from .public_reference_supervision import patch_world_windows


def named_public_polygons(objects,name,expected_type=None):
    legal={'Road':{'TrafficRoad','Footpath','Bridge'},'Building':{'Building'}}
    ids=[];contours=[];invalid=[];mismatch=[]
    for oid,obj in sorted(objects.items(),key=lambda x:str(x[0])):
        if canonical_name(obj.get('name',''))!=canonical_name(name):continue
        if expected_type and obj['object_type'] not in legal.get(expected_type,{expected_type}):
            mismatch.append(str(oid));continue
        polygon=shapely.Polygon(obj['contour'])
        if not polygon.is_valid or polygon.area<=0:invalid.append(str(oid));continue
        ids.append(str(oid));contours.append(obj['contour'])
    return dict(ids=ids,contours=contours,invalid_ids=invalid,type_mismatch_ids=mismatch)


def _polygons(geometry):
    if geometry.is_empty:return []
    if geometry.geom_type=='Polygon':return [geometry]
    return [p for part in geometry.geoms for p in _polygons(part)] if hasattr(geometry,'geoms') else []


def _mask(geometry,footprint):
    result=np.zeros((384,384),np.uint8)
    for poly in _polygons(geometry):
        px=world_to_frame(np.asarray(poly.exterior.coords),footprint,result.shape)
        clipped=shapely.Polygon(px).intersection(shapely.box(0,0,383,383))
        for piece in _polygons(clipped):
            cv2.fillPoly(result,[np.rint(np.asarray(piece.exterior.coords)).astype(np.int32)],1)
            for hole in piece.interiors:cv2.fillPoly(result,[np.rint(np.asarray(hole.coords)).astype(np.int32)],0)
        for hole in poly.interiors:
            hp=world_to_frame(np.asarray(hole.coords),footprint,result.shape)
            cv2.fillPoly(result,[np.rint(hp).astype(np.int32)],0)
    return result.astype(bool)


def native_patch_validity(rgb,depth):
    support=(rgb.max(-1)>0)&np.asarray(depth,bool)
    return (support.reshape(24,16,24,16).mean((1,3))>=.9).ravel()[SPARSE_IDS]


def local_landmark_anchor(contours,observation,*,patch_valid=None,margin_m=.5,min_pixels=256,min_positive=2):
    rgb=np.asarray(observation['bgr']);depth=np.asarray(observation['depth_valid'],bool);fp=np.asarray(observation['footprint_world'])
    if rgb.shape!=(384,384,3) or depth.shape!=(384,384):raise ValueError('Native384 current RGB/depth required')
    region=named_union(contours);view=shapely.Polygon(fp);local=region.intersection(view)
    geometry_mask=_mask(local,fp)
    # Erode the PUBLIC object boundary, never the artificial camera clipping edge.
    interior=region.buffer(-margin_m).intersection(view)
    stable_mask=_mask(interior,fp);source=(rgb.max(-1)>0)&depth
    valid_mask=stable_mask&source
    valid=native_patch_validity(rgb,depth)
    if patch_valid is not None:
        external=np.asarray(patch_valid,bool)
        if external.shape!=valid.shape:raise ValueError('144 aligned patch flags required')
        valid &= external
    labels,coverage=weak_region_labels(patch_world_windows(fp)[:,0],region,valid,margin=margin_m)
    positives=int((labels==1).sum());negatives=int((labels==0).sum())
    pixel_usable=int(valid_mask.sum())>=min_pixels
    return dict(local_geometry=local,whole_fraction=local.area/region.area,view_fraction=local.area/view.area,
        geometry_mask=geometry_mask,valid_mask=valid_mask,source_mask=geometry_mask&source,
        whole_pixels=int(geometry_mask.sum()),stable_valid_pixels=int(valid_mask.sum()),
        local_source_support=float((geometry_mask&source).sum()/max(int(geometry_mask.sum()),1)),
        pixel_usable=pixel_usable,train_eligible=bool(pixel_usable and positives>=min_positive),
        labels=labels,coverage=coverage,valid_patches=valid,positive_patches=positives,negative_patches=negatives,
        positive_only=bool(positives>=min_positive and negatives==0),actual_visibility='unknown',pixel_ground_truth=False)


def local_anchor_tokens(contours,footprint,vertices=16):
    local=named_union(contours).intersection(shapely.Polygon(footprint))
    if local.is_empty or local.area<=0:raise ValueError('No local observed Polygon')
    boundary=local.boundary
    points=np.asarray([boundary.interpolate(t,normalized=True).coords[0] for t in np.arange(vertices)/vertices])
    pixels=np.clip(world_to_frame(points,footprint,(384,384,3))/383.,0,1)
    # Public registered coordinates are legitimate inputs; DO NOT re-center.
    return np.concatenate((pixels,np.roll(pixels,-1,axis=0)-pixels),-1).astype(np.float32)
