"""Projection-only diagnostics. No model, training, GT target or policy changes."""
import cv2
import numpy as np
import shapely
from .landmark_projection import world_to_frame,frame_to_world,project_landmark
from .observed_landmarks import current_landmark_rois
from .native_dense_regions import native_dense_regions,sparse_view
from .public_reference_supervision import patch_world_windows
from .landmark_region_supervision import weak_region_labels


def _fill_geometry(mask,geometry):
    if geometry.is_empty:return
    if geometry.geom_type=='Polygon':
        cv2.fillPoly(mask,[np.rint(np.asarray(geometry.exterior.coords)).astype(np.int32)],1)
        for hole in geometry.interiors:cv2.fillPoly(mask,[np.rint(np.asarray(hole.coords)).astype(np.int32)],0)
    elif hasattr(geometry,'geoms'):
        for sub in geometry.geoms:_fill_geometry(mask,sub)


def audit_projection(contour,footprint,rgb,depth_valid):
    contour=np.asarray(contour,float);footprint=np.asarray(footprint,float)
    region=shapely.Polygon(contour);view=shapely.Polygon(footprint)
    if not region.is_valid or region.area<=0 or not view.is_valid or view.area<=0:raise ValueError('Valid complete world polygons required')
    h,w=rgb.shape[:2]
    if (h,w)!=(384,384) or np.asarray(depth_valid).shape!=(h,w):raise ValueError('Actual native384 observation required')
    intersection=region.intersection(view);px=world_to_frame(contour,footprint,rgb.shape)
    image_polygon=shapely.Polygon(px);clip=image_polygon.intersection(shapely.box(0,0,w-1,h-1))
    mask=np.zeros((h,w),np.uint8);_fill_geometry(mask,clip)
    rgb_valid=rgb.max(-1)>0;source_valid_mask=mask.astype(bool)&rgb_valid&depth_valid
    pixel_roundtrip=world_to_frame(frame_to_world(px,footprint,rgb.shape),footprint,rgb.shape)
    landmark=dict(id='audit',name='public polygon',object_type='Building',contour=contour)
    thresholds={}
    for name,fraction,pixels in [('default50_32',.5,32),('roi50_256',.5,256),('roi90_256',.9,256)]:
        thresholds[name]=project_landmark(landmark,footprint,rgb.shape,min_visible=fraction,min_pixels=pixels) is not None
    current=current_landmark_rois(dict(bgr=rgb,depth_valid=depth_valid,footprint_world=footprint),[landmark])
    dense=sparse_view(native_dense_regions(dict(bgr=rgb,depth_valid=depth_valid,footprint_world=footprint)))
    label,coverage=weak_region_labels(patch_world_windows(footprint)[:,0],region,dense['valid'])
    old=np.zeros((h,w),np.uint8);cv2.fillPoly(old,[np.rint(px).astype(np.int32)],1)
    area=float(abs(cv2.contourArea(px)))
    return dict(geometry=dict(polygon_area_m2=region.area,view_area_m2=view.area,intersection_m2=intersection.area,
        polygon_coverage=intersection.area/region.area,view_occupancy=intersection.area/view.area,
        polygon_bigger_than_view=region.area>view.area,world_distance_to_view_m=region.distance(view)),
        pixel_polygon=px.tolist(),mask=mask,source_valid_mask=source_valid_mask.astype(np.uint8),
        projected_area_px=area,projected_mask_pixels=int(mask.sum()),
        legacy_fraction=min(1.,float(old.sum()/max(area,1e-12))),
        clipped_vs_legacy_disagreement_pixels=int((mask!=old).sum()),
        rgb_support_fraction=float(rgb_valid[mask.astype(bool)].mean()) if mask.any() else None,
        depth_support_fraction=float(depth_valid[mask.astype(bool)].mean()) if mask.any() else None,
        valid_observed_region_pixels=int(source_valid_mask.sum()),roundtrip_max_px=float(np.linalg.norm(px-pixel_roundtrip,axis=-1).max()),
        thresholds=dict(**thresholds,current_roi90_pipeline=bool(current)),
        patches=dict(total=144,valid90=int(dense['valid'].sum()),positive90=int((label==1).sum()),negative=int((label==0).sum()),unknown=int((label<0).sum()),
            inner_coverage50=int(((coverage[:,0]>=.5)&dense['valid']).sum()),
            pgvb_cache_eligible_geometric=bool((label==1).sum()>=2 and (label==0).sum()>=4)),
        actual_visibility='unknown',human_pixel_gt=False)


def raster_projection_discrepancy(footprint,transform,size=384):
    """Compare advertised projection to the renderer's integer raster-index warp.

    Pixel-center convention assumes the orthophoto PNG and TIF share sample
    centers. Edge convention is reported separately; neither is pixel GT.
    """
    footprint=np.asarray(footprint,float);inverse=~transform
    source=np.column_stack((inverse.a*footprint[:,0]+inverse.b*footprint[:,1]+inverse.c,inverse.d*footprint[:,0]+inverse.e*footprint[:,1]+inverse.f));rounded=np.floor(source)
    dst=np.float32([[0,0],[size-1,0],[size-1,size-1],[0,size-1]])
    matrix=cv2.getPerspectiveTransform(rounded.astype(np.float32),dst)
    yy,xx=np.meshgrid(np.linspace(0,size-1,9),np.linspace(0,size-1,9),indexing='ij')
    desired=np.stack((xx.ravel(),yy.ravel()),-1).astype(np.float32)
    world=frame_to_world(desired,footprint,(size,size))
    sample=np.column_stack((inverse.a*world[:,0]+inverse.b*world[:,1]+inverse.c,inverse.d*world[:,0]+inverse.e*world[:,1]+inverse.f)).astype(np.float32)
    def error(offset):
        projected=cv2.perspectiveTransform((sample-offset)[None],matrix)[0]
        return np.linalg.norm(projected-desired,axis=-1)
    center=error(.5);edge=error(0.)
    centers=rounded+.5
    effective=np.column_stack((transform.a*centers[:,0]+transform.b*centers[:,1]+transform.c,transform.d*centers[:,0]+transform.e*centers[:,1]+transform.f))
    expected_edge=footprint[1]-footprint[0];actual_edge=effective[1]-effective[0]
    angle=lambda xy:np.degrees(np.arctan2(xy[1],xy[0]))
    return dict(max_center_convention_px=float(center.max()),mean_center_convention_px=float(center.mean()),
        max_edge_convention_px=float(edge.max()),mean_edge_convention_px=float(edge.mean()),
        corner_world_shift_max_m=float(np.linalg.norm(effective-footprint,axis=-1).max()),
        first_edge_length_ratio=float(np.linalg.norm(actual_edge)/np.linalg.norm(expected_edge)),
        first_edge_angle_difference_deg=float((angle(actual_edge)-angle(expected_edge)+180)%360-180),
        raster_source_corners=source.tolist(),integer_source_corners=rounded.astype(int).tolist(),
        effective_footprint_pixel_centers=effective.tolist(),source_pixel_scale_m=[float(abs(transform.a)),float(abs(transform.e))])
