"""Post-hoc visibility/clarity PROXIES for V2 NMS5 hits; no feedback to navigation.
Target identity/contour/coordinates only score already recorded/current start observations.
No actual visibility or instance accuracy without independent pixel verification.
"""
import argparse,json,sys
from pathlib import Path
from collections import Counter
import numpy as np,cv2
from shapely.geometry import Polygon,Point
from graphnav.renderer import CityNavObservedViewRenderer
from graphnav.landmark_projection import world_to_frame


def score(ep,poses,obj,renderer,ground):
    center=np.array(ep.target_position[:2]);contour=np.array(obj.get('contour',[]));poly=Polygon(contour) if contour.ndim==2 and len(contour)>=3 else None
    if poly is not None and (not poly.is_valid or poly.area<=0):poly=None
    mismatch=poly is not None and poly.distance(Point(center))>20
    results=[]
    for step,pose in enumerate(poses):
        obs=renderer.current_rgbd(ep.map_name,np.asarray(pose),ground);foot=Polygon(obs['footprint_world']);r=dict(step=step,center_in_footprint=foot.covers(Point(center)),polygon_available=poly is not None,center_contour_disagreement_gt20m=bool(mismatch),geometric_overlap=False,valid_roi=False,pixel_area=0,min_bbox_side_px=0.,clarity_proxy_8=False,clarity_proxy_16=False,clarity_proxy_32=False)
        if poly is not None:
            clipped=poly.intersection(foot);r['geometric_overlap']=not clipped.is_empty and clipped.area>0
            if r['geometric_overlap']:
                mask=np.zeros(obs['bgr'].shape[:2],np.uint8)
                for g in getattr(clipped,'geoms',[clipped]):
                    if g.geom_type!='Polygon':continue
                    px=world_to_frame(np.array(g.exterior.coords),obs['footprint_world'],obs['bgr'].shape);cv2.fillPoly(mask,[np.rint(px).astype(np.int32)],1)
                    for hole in g.interiors:
                        hx=world_to_frame(np.array(hole.coords),obs['footprint_world'],obs['bgr'].shape);cv2.fillPoly(mask,[np.rint(hx).astype(np.int32)],0)
                support=(mask>0)&obs['depth_valid']&(obs['bgr'].max(-1)>0);yy,xx=np.where(support);area=len(xx);validfrac=area/max(int(mask.sum()),1);side=float(min(np.ptp(xx)+1,np.ptp(yy)+1)) if area else 0.;r.update(pixel_area=area,min_bbox_side_px=side,valid_pixel_fraction=validfrac,polygon_fraction_in_footprint=float(clipped.area/poly.area),valid_roi=bool(area>0 and not mismatch))
                for k in [8,16,32]:r['clarity_proxy_'+str(k)]=bool(not mismatch and validfrac>=.8 and side>=k and area>=k*k)
        results.append(r)
    return results


def main():
    p=argparse.ArgumentParser()
    for k in ['data','sbf-source','initial-eval','navigation','output']:p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--initial-only',action='store_true');a=p.parse_args();sys.path.insert(0,str(a.sbf_source));from sbfnav.dataset import CityNavDataset
    from sbfnav.mapdata import GROUND_LEVEL
    ds=CityNavDataset(a.data,'val_seen',version='processed_citynav');targets=json.load(open(a.data/'cityrefer/objects.json'));rows=json.load(open(a.initial_eval/'original_v2_rows.json'));selected=[r for r in rows if r['nms_hits']['5']];assert len(selected)==1911
    renderer=CityNavObservedViewRenderer(a.data/'rgbd',a.output/'observations',image_size=384);a.output.mkdir(parents=True,exist_ok=True);output=[]
    if not a.initial_only:
        metrics=json.load(open(a.navigation/'original_v2/metrics.json'));assert metrics['complete'] and metrics['coverage']['completed']==2470
    for row in selected:
        i=row['index'];ep=ds[i];obj=targets.get(ep.map_name,{}).get(str(ep.object_id),{})
        if a.initial_only:poses=[ep.start_pose]
        else:
            nav=json.load(open(a.navigation/'original_v2/rows'/f'{i:05d}.json'));poses=nav['poses']
            # rollout does not observe the final moved pose after horizon/stagnation; policy_stop does observe it.
            poses=poses[:len(nav['diagnostics'])]
        frames=score(ep,poses,obj,renderer,GROUND_LEVEL[ep.map_name]);output.append(dict(index=i,episode=row['episode'],map_name=ep.map_name,category=obj.get('object_type','unknown'),frames=frames))
        if len(output)%200==0:print('scored',len(output),flush=True)
    keys=['center_in_footprint','geometric_overlap','valid_roi','clarity_proxy_8','clarity_proxy_16','clarity_proxy_32','center_contour_disagreement_gt20m']
    def counts(rr):return dict(n=len(rr),initial={k:sum(bool(r['frames'] and r['frames'][0][k]) for r in rr) for k in keys},any_legal_observation={k:sum(any(f[k] for f in r['frames']) for r in rr) for k in keys})
    scope='initial' if a.initial_only else 'causal_history';summary=dict(scope=scope,**counts(output),by_category={c:counts([r for r in output if r['category']==c]) for c in sorted({r['category'] for r in output})},native_observation_px=384,clarity_definition='Projected contour has >=80% valid RGB/depth, bbox short side>=k, supported pixel area>=k^2; NOT verified visibility/recognition',verified_clear_target_count=None,no_independent_pixel_truth=True,GT_usage='post-hoc masks/coverage/size only, never policy or view selection',causal_note='History uses actually observed pre-decision poses only; excludes unobserved terminal moves at horizon/stagnation; no teacher trajectory.')
    (a.output/(scope+'_rows.json')).write_text(json.dumps(output,separators=(',',':')));(a.output/(scope+'_summary.json')).write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2),flush=True)
if __name__=='__main__':main()
