"""48 public TRAIN polygon/native384 projections; never load a model or target labels."""
import argparse,collections,hashlib,json,math,sys,time
from pathlib import Path
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import rasterio
import shapely
from graphnav.checkpointing import atomic_json,file_sha256
from graphnav.landmark_alias import canonical_name
from graphnav.memory import ObservedRGBMemory
from graphnav.renderer import CityNavObservedViewRenderer
from graphnav.polygon_projection_audit import audit_projection,raster_projection_discrepancy
from scripts.analyze_instruction_landmark_association import GENERIC_NAMES


def stratum(region,view):
    overlap=region.intersection(view).area;fraction=overlap/region.area;occupancy=overlap/view.area
    if fraction>=.98:return 'full'
    if region.area>view.area*1.25 and occupancy>=.1:return 'oversized'
    if .05<=fraction<.95 and occupancy<.8:return 'partial'
    if overlap==0 and 1<=region.distance(view)<=60:return 'outside'
    return None


def select_cases(samples,catalog,ground):
    bymap=collections.defaultdict(dict)
    for r in samples:
        if r['frame']['valid_fraction']>=.4:bymap[r['map_name']].setdefault(r['frame_key'],r)
    candidates=collections.defaultdict(list)
    for mp,values in sorted(bymap.items()):
        poses=sorted(values.values(),key=lambda r:hashlib.sha256(r['frame_key'].encode()).hexdigest())[:4]
        objects=[]
        for oid,obj in catalog[mp].items():
            name=canonical_name(obj.get('name',''));typ=obj['object_type']
            typ='Road' if typ in ('TrafficRoad','Footpath','Bridge') else typ
            if typ not in ('Building','Road') or len(name)<3 or name in GENERIC_NAMES:continue
            poly=shapely.Polygon(obj['contour'])
            if poly.is_valid and poly.area>4:objects.append((oid,obj,typ,poly))
        for original in poses:
            for height in (15,35,60,100):
                for yawdeg in (0,45,90,135,180,225,270,315):
                    xyz=original['frame']['camera_pose'][:3];pose=[xyz[0],xyz[1],ground[mp]+height,math.radians(yawdeg)]
                    footprint=ObservedRGBMemory.view_corners(pose[:2],pose[3],height);view=shapely.Polygon(footprint)
                    for oid,obj,typ,poly in objects:
                        state=stratum(poly,view)
                        if state is None:continue
                        if state!='outside' and poly.intersection(view).area/view.area*384**2<256:continue
                        seed=f'{mp}:{oid}:{original["frame_key"]}:{height}:{yawdeg}'
                        candidates[(typ,state)].append(dict(map_name=mp,public_id=str(oid),public_name=obj['name'],source_type=obj['object_type'],type=typ,
                            contour=obj['contour'],source_frame_key=original['frame_key'],source_image_sha256=original['frame']['image_sha256'],
                            source_camera_pose=original['frame']['camera_pose'],pose=pose,height=height,yaw_deg=yawdeg,state=state,rank=hashlib.sha256(seed.encode()).hexdigest()))
    print(json.dumps({'selection_pool_counts':{str(k):len(v) for k,v in candidates.items()}}),flush=True)
    chosen=[];identities=set();map_use=collections.Counter();yaw_use=collections.Counter();height_use=collections.Counter()
    for typ in ('Building','Road'):
        for state in ('full','partial','oversized','outside'):
            pool=candidates[(typ,state)]
            for _ in range(6):
                options=[r for r in pool if (typ,state,r['map_name'],r['public_id']) not in identities]
                if not options:raise ValueError('Insufficient distinct public polygons for '+typ+'/'+state)
                options.sort(key=lambda r:(map_use[r['map_name']],yaw_use[r['yaw_deg']],height_use[r['height']],r['rank']))
                pick=options[0];chosen.append(pick);identities.add((typ,state,pick['map_name'],pick['public_id']));map_use[pick['map_name']]+=1;yaw_use[pick['yaw_deg']]+=1;height_use[pick['height']]+=1
    return sorted(chosen,key=lambda r:(r['map_name'],r['rank'])),{str(k):len(v) for k,v in candidates.items()}


def main():
    p=argparse.ArgumentParser()
    for k in ('source-data','catalog','data-root','sbf-source','output'):p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError('Preserve previous audit outputs')
    a.output.mkdir(parents=True);sys.path.insert(0,str(a.sbf_source));from sbfnav.mapdata import GROUND_LEVEL
    proof=json.loads((a.source_data/'manifest.json').read_text());assert file_sha256(a.source_data/'samples_public.json')==proof['samples_sha256']
    # Existing source is strictly TRAIN; only public frame metadata is used.
    for path,h in proof['source_manifests'].items():
        assert file_sha256(path)==h
        m=json.loads(Path(path).read_text());assert m['split']=='train_seen'
    samples=json.loads((a.source_data/'samples_public.json').read_text());catalog=json.loads(a.catalog.read_text())
    chosen,pool_counts=select_cases(samples,catalog,GROUND_LEVEL);atomic_json(chosen,a.output/'selection.json')
    renderer=CityNavObservedViewRenderer(a.data_root/'rgbd',a.output/'unused',image_size=384)
    records=[];started=time.perf_counter();contacts=[];rasters={};pixel_conventions=[]
    for i,case in enumerate(chosen):
        folder=a.output/f'case_{i:02d}';folder.mkdir();mp=case['map_name'];pose=np.asarray(case['pose'])
        obs=renderer.current_rgbd(mp,pose,GROUND_LEVEL[mp]);bgr=obs['bgr'];footprint=obs['footprint_world']
        png=folder/'rgb.png';cv2.imwrite(str(png),bgr)
        result=audit_projection(np.asarray(case['contour']),footprint,bgr,obs['depth_valid'])
        mask=result.pop('mask');source_mask=result.pop('source_valid_mask')
        cv2.imwrite(str(folder/'geometric_mask.png'),mask*255);cv2.imwrite(str(folder/'source_valid_mask.png'),source_mask*255)
        cv2.imwrite(str(folder/'depth_valid.png'),obs['depth_valid'].astype(np.uint8)*255)
        if mp not in rasters:
            with rasterio.open(a.data_root/'rgbd'/f'{mp}.tif') as raster:
                rasters[mp]=(raster.transform,raster.width,raster.height)
        transform,width,height=rasters[mp]
        assert renderer._rgb.shape[:2]==(height,width),'PNG/TIF source shapes disagree'
        discrepancy=raster_projection_discrepancy(footprint,transform)
        metrics=dict(index=i,**case,**result,footprint_world=footprint.tolist(),raster_warp=discrepancy,
            actual_rgb_shape=list(bgr.shape),rgb_sha256=file_sha256(png),geometric_mask_sha256=file_sha256(folder/'geometric_mask.png'),
            re_render='controlled yaw/height at recorded TRAIN XY; not an original human trajectory frame',
            human_review=None,AI_visual_review=None)
        atomic_json(metrics,folder/'sample.json');records.append(metrics)
        overlay=bgr.copy();px=np.asarray(result['pixel_polygon']);cv2.polylines(overlay,[np.rint(px).astype(np.int32)],True,(0,255,255),2)
        colored=np.zeros_like(bgr);colored[mask.astype(bool)]=(130,130,130);colored[source_mask.astype(bool)]=(0,220,0)
        cv2.imwrite(str(folder/'overlay.png'),overlay);cv2.imwrite(str(folder/'support_mask.png'),colored)
        trip=np.concatenate((bgr,overlay,colored),axis=1);cv2.imwrite(str(folder/'comparison.png'),trip)
        fig,axes=plt.subplots(1,3,figsize=(12,4.5))
        for ax,img,title in zip(axes,(bgr,overlay,colored),('Actual RGB 384px','Projected full Polygon','Grey: clipped geometry / Green: valid source')):
            ax.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB));ax.set_title(title,fontsize=9);ax.axis('off')
        g=result['geometry'];fig.suptitle(f'{i:02d} {case["public_name"]} | {case["type"]}/{case["state"]}\nyaw={case["yaw_deg"]} deg, AGL={case["height"]}m | polygon coverage={g["polygon_coverage"]:.3f}, view occupancy={g["view_occupancy"]:.3f}',fontsize=10)
        fig.tight_layout();fig.savefig(folder/'figure.png',dpi=120);plt.close(fig)
        thumbnail=cv2.resize(trip,(768,256));header=np.full((54,768,3),255,np.uint8)
        cv2.putText(header,f'{i:02d} {case["type"]}/{case["state"]} yaw {case["yaw_deg"]} h {case["height"]}',(8,20),cv2.FONT_HERSHEY_SIMPLEX,.5,(0,0,0),1)
        cv2.putText(header,f'{case["public_name"][:55]} | full {g["polygon_coverage"]:.3f} view {g["view_occupancy"]:.3f}',(8,43),cv2.FONT_HERSHEY_SIMPLEX,.45,(0,0,0),1)
        contacts.append(np.vstack((header,thumbnail)))
        print(json.dumps(dict(completed=i+1,total=48,map_name=mp,name=case['public_name'],state=case['state'])),flush=True)
    for page in range(4):
        block=contacts[page*12:(page+1)*12];sheet=np.vstack([np.hstack(block[j:j+2]) for j in range(0,12,2)])
        cv2.imwrite(str(a.output/f'contact_{page}.png'),sheet)
    def summarize(rr):
        observed=[r for r in rr if r['valid_observed_region_pixels']>=256]
        return dict(n=len(rr),geometric_overlap=sum(r['geometry']['intersection_m2']>0 for r in rr),
            valid_observed_geometric_region_ge256=len(observed),
            threshold_pass={k:sum(r['thresholds'][k] for r in rr) for k in ('default50_32','roi50_256','roi90_256','current_roi90_pipeline')},
            supported_but_rejected50=sum(not r['thresholds']['roi50_256'] for r in observed),
            supported_but_rejected90=sum(not r['thresholds']['roi90_256'] for r in observed),
            patches_pgvb_eligible=sum(r['patches']['pgvb_cache_eligible_geometric'] for r in rr))
    summary=dict(samples=len(records),source_polygon_fragments=len({(r['map_name'],r['public_id']) for r in records}),
        map_names=sorted({r['map_name'] for r in records}),same_name_city_groups=len({(r['map_name'].split('_block_')[0],canonical_name(r['public_name'])) for r in records}),
        counts={f'{t}/{s}':summarize([r for r in records if r['type']==t and r['state']==s]) for t in ('Building','Road') for s in ('full','partial','oversized','outside')},
        overall=summarize(records),yaw_counts=dict(collections.Counter(r['yaw_deg'] for r in records)),height_counts=dict(collections.Counter(r['height'] for r in records)),
        analytic_roundtrip_max_px=max(r['roundtrip_max_px'] for r in records),
        warp_center_max_px=max(r['raster_warp']['max_center_convention_px'] for r in records),
        warp_center_median_sample_max_px=float(np.median([r['raster_warp']['max_center_convention_px'] for r in records])),
        warp_max_world_corner_shift_m=max(r['raster_warp']['corner_world_shift_max_m'] for r in records),
        max_edge_angle_difference_deg=max(abs(r['raster_warp']['first_edge_angle_difference_deg']) for r in records),
        max_edge_length_ratio_error=max(abs(r['raster_warp']['first_edge_length_ratio']-1) for r in records),
        sourcePNG_TIF_shape_matches=True,seconds=time.perf_counter()-started,
        source_samples_sha256=file_sha256(a.source_data/'samples_public.json'),catalog_sha256=file_sha256(a.catalog),
        source_code_sha256={str(p):file_sha256(p) for p in (Path(__file__),Path('graphnav/polygon_projection_audit.py'),Path('graphnav/renderer.py'),Path('graphnav/landmark_projection.py'),Path('graphnav/observed_landmarks.py'),Path('graphnav/native_dense_regions.py'),Path('graphnav/landmark_region_supervision.py'))},
        no_model_loaded=True,no_training=True,no_navigation=True,actual_visibility_verified=False,human_masks=0,
        warning='Balanced controlled TRAIN stress cases, not prevalence. Geometric projection is not actual visibility/semantic segmentation. No human pixel IoU available.')
    atomic_json(records,a.output/'samples.json');atomic_json(summary,a.output/'summary.json')
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
