"""Build observed-only CityRefer region data from preserved CityFlight pose frames."""
import argparse,hashlib,json,sys,time
from pathlib import Path
from collections import Counter,defaultdict
import cv2
import numpy as np
from PIL import Image,ImageDraw
from shapely.geometry import Polygon
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from graphnav.landmark_projection import project_landmark,make_roi_image
from experiments.visual_grounding.data.audit_attributes import object_attributes,bind_phrase
from experiments.visual_grounding.data.fast_projection import project_validated

def main():
    p=argparse.ArgumentParser();p.add_argument('--data-root',default='/home/rental/20260922_1/Workspace/DATA/sbfnav-data');p.add_argument('--observations-root',default='/home/rental/20260922_1/Workspace/Ablations/GraphNav/2027-CVPR-bidirectional/data/p0');p.add_argument('--output',default='runs/visual_grounding_benchmark/data_audit');p.add_argument('--types',nargs='*',default=[]);p.add_argument('--splits',nargs='+',default=['train_seen','val_seen','val_unseen']);p.add_argument('--map');a=p.parse_args()
    out=Path(a.output).resolve();out.mkdir(parents=True,exist_ok=True);(out/'rois').mkdir(exist_ok=True);dr=Path(a.data_root);obsroot=Path(a.observations_root)
    objects=json.loads((dr/'cityrefer/objects.json').read_text());processed=json.loads((dr/'cityrefer/processed_descriptions.json').read_text())
    catalog={};attrs={};geom=Counter()
    for m,oo in objects.items():
        catalog[m]=[]
        for oid,o in oo.items():
            coords=np.asarray(o['contour'],np.float64);poly=Polygon(coords)
            attrs[m,oid]=object_attributes(processed.get(m,{}).get(oid,[]),o['object_type'])
            if not poly.is_valid or poly.area<=0:geom[o['object_type']+'_invalid']+=1;continue
            geom[o['object_type']+'_valid']+=1
            if not a.types or o['object_type'] in a.types:catalog[m].append((oid,o,coords.min(0),coords.max(0)))
    (out/'objects.json').write_text(json.dumps(objects))
    bounds={m:(np.asarray([r[2] for r in rows]),np.asarray([r[3] for r in rows])) for m,rows in catalog.items()}
    cv2.setNumThreads(1)
    stats={'geometry':dict(geom),'protocol':{'source':'preserved actual CityFlight trajectory pose frames rendered from RGB raster','roi':'geometric weak polygon; not verified segmentation','labels':'consistent CityRefer processed target language weak labels; not image verified','future_protection':'every ROI source step <= same episode query cutoff','source_frame_selection':'all episodes fixed 0/50%/80% human prefix steps' if 'full_observations' in str(obsroot) else 'previously frozen map-stratified stride8 maxstep64 observations; no new GT-guided views'},'splits':{}}
    samples=[];regions=[];start=time.monotonic();materialized_views=defaultdict(set);projection_checks=0
    with (out/'regions.jsonl').open('w') as rf,(out/'manifest.jsonl').open('w') as mf:
      for split in a.splits:
        raw=json.loads((dr/f'processed_citynav/citynav_{split}.json').read_text());observations=[json.loads(l) for l in (obsroot/f'observations_{split}.jsonl').open()];observations=[r for r in observations if not a.map or r['map_name']==a.map];effective_raw=[r for r in raw if not a.map or f"{r['area']}_block_{r['block']}"==a.map];count=Counter();types=Counter();attrcounts=Counter();instances=set();target_instances=set();episodes=set()
        for oi,obs in enumerate(observations):
            idx=obs['source_episode_index'];episode=raw[idx];m=obs['map_name'];target=str(episode['object_ids'][0]);did=episode['ann_ids'][0];cutoff=obs['observation_cutoff_step'];instruction=episode['descriptions'][0];episode_id=f'{split}:{idx}:{m}:{target}:{did}'
            for frame in obs['rgb_observations']:
                step=frame['step'];assert step<=cutoff
                path=Path(frame['path']);path=path if path.exists() else obsroot/'frames'/split/m/f'source_{idx}'/path.name
                image=cv2.imread(str(path));assert image is not None and image.shape==(384,384,3)
                footprint=np.asarray(frame['footprint_world'],np.float32);lo=footprint.min(0);hi=footprint.max(0)
                target_object=objects[m][target];ds=processed.get(m,{}).get(target,[]);target_phrase=ds[did].get('target','') if did<len(ds) else '';target_attr=bind_phrase(target_phrase,target_object['object_type'])
                target_probe=project_validated(target_object,footprint,image.shape) if Polygon(target_object['contour']).is_valid else None
                target_match_frame=False
                if target_probe is not None and target_attr:
                    tx0,ty0,tx1,ty1=target_probe.bbox_xyxy
                    target_match_frame=min(tx1-tx0,ty1-ty0)>=24 and float((image.max(2)>0)[target_probe.mask.astype(bool)].mean())>=.9
                footprint_key=hashlib.sha256(np.round(footprint,3).tobytes()).hexdigest()
                lows,highs=bounds[m]
                overlaps=np.flatnonzero(np.all(highs>=lo,axis=1)&np.all(lows<=hi,axis=1))
                for ci in overlaps:
                    oid,o,olo,ohi=catalog[m][ci]
                    count['bbox_intersection']+=1
                    roi=project_validated(o,footprint,image.shape,min_visible=.9,min_pixels=256)
                    if projection_checks<500:
                        original=project_landmark(o,footprint,image.shape,min_visible=.9,min_pixels=256)
                        assert (roi is None)==(original is None)
                        if roi is not None:
                            assert roi.bbox_xyxy==original.bbox_xyxy and np.array_equal(roi.mask,original.mask) and np.array_equal(roi.contour_px,original.contour_px) and roi.visible_fraction==original.visible_fraction
                        projection_checks+=1
                    if roi is None:count['projection_small_clipped_or_invalid']+=1;continue
                    x0,y0,x1,y1=roi.bbox_xyxy
                    if min(x1-x0,y1-y0)<8:count['bbox_below8']+=1;continue
                    support=float((image.max(2)>0)[roi.mask.astype(bool)].mean())
                    if support<.9:count['source_rgb_holes']+=1;continue
                    ident=f'{split}:{idx}:{step}:{m}:{oid}';sid=hashlib.sha256(ident.encode()).hexdigest()[:20];rp=out/'rois'/f'{sid}.png'
                    viewkey=(split,m,oid);materialize=(min(x1-x0,y1-y0)>=24 and ((footprint_key not in materialized_views[viewkey] and len(materialized_views[viewkey])<2) or (target_match_frame and o['object_type']==target_object['object_type'] and attrs[m,oid])))
                    if materialize:
                        if not rp.exists():cv2.imwrite(str(rp),make_roi_image(image,roi,output_size=min(384,max(32,int(max(x1-x0,y1-y0)*1.5)))),[cv2.IMWRITE_PNG_COMPRESSION,1])
                        materialized_views[viewkey].add(footprint_key);count['materialized_rois']+=1
                    quality={'geometry_visible_fraction':roi.visible_fraction,'rgb_source_support':support,'native_bbox_min_side':min(x1-x0,y1-y0),'attribute_resolution_eligible':min(x1-x0,y1-y0)>=24,'semantic_boundary_verified':False}
                    row=dict(sample_id=sid,map_name=m,split=split,episode_id=episode_id,source_episode_index=idx,instruction=instruction if oid==target else '',attribute_phrase='',target_object_id=target,object_id=oid,object_type=o['object_type'],object_name=o.get('name',''),is_episode_target=oid==target,observed_step=step,observation_cutoff_step=cutoff,roi_path=str(rp) if materialize else None,roi_source=str(path.resolve()),roi_bbox=list(roi.bbox_xyxy),roi_visibility=roi.visible_fraction,roi_pixel_size=[x1-x0,y1-y0],roi_quality=quality,annotation_confidence='geometric_weak_roi_not_semantic_segmentation',attributes=attrs[m,oid],footprint_world=frame['footprint_world'],object_geometry_key=f'{m}:{oid}',object_center_world=np.asarray(o['contour']).mean(0).tolist(),footprint_key=footprint_key)
                    rf.write(json.dumps(row)+'\n');regions.append(row);count['valid_rois']+=1;types[o['object_type']]+=1;instances.add((m,oid));episodes.add(idx)
                    if quality['attribute_resolution_eligible']:
                        for k in attrs[m,oid]:attrcounts[k]+=1
                    if oid==target:
                        ds=processed.get(m,{}).get(oid,[]);phrase=ds[did].get('target','') if did<len(ds) else ''
                        row=dict(row,attribute_phrase=phrase,query_attributes=bind_phrase(phrase,o['object_type']))
                        mf.write(json.dumps(row)+'\n');samples.append(row);count['target_rois']+=1;target_instances.add((m,oid))
            if oi%500==0:rf.flush();mf.flush();print(json.dumps({'split':split,'frames':oi,'regions':count['valid_rois'],'seconds':time.monotonic()-start}),flush=True)
        stats['splits'][split]=dict(episodes_total=len(effective_raw),unique_instructions=len({(f"{r['area']}_block_{r['block']}",r['descriptions'][0]) for r in effective_raw}),unique_target_objects=len({(f"{r['area']}_block_{r['block']}",r['object_ids'][0]) for r in effective_raw}),maps=len({f"{r['area']}_block_{r['block']}" for r in effective_raw}),observations=len(observations),observed_episode_count=len(episodes),counts=dict(count),roi_types=dict(types),roi_objects=len(instances),visible_target_objects=len(target_instances),weak_attribute_roi_counts=dict(attrcounts),verified_attribute_samples=0)
    # Fixed representative sample, including targets and all object categories.
    eligible=[r for r in regions if r['split']!='val_unseen' and r['roi_path'] and r['roi_quality']['attribute_resolution_eligible']];rng=np.random.default_rng(20261010);choices=list(rng.choice(len(eligible),min(150,len(eligible)),replace=False));cases=[eligible[i] for i in choices];(out/'cases').mkdir(exist_ok=True)
    for si in range(0,len(cases),12):
        sheet=Image.new('RGB',(1000,870),'white');draw=ImageDraw.Draw(sheet)
        for j,r in enumerate(cases[si:si+12]):
            x=(j%4)*250;y=(j//4)*290;im=Image.open(r['roi_path']).convert('RGB');im.thumbnail((246,240));sheet.paste(im,(x,y));label=f"{si+j} {r['map_name']} obj{r['object_id']}\n{r['object_type']} target={r['is_episode_target']}\nep{r['source_episode_index']} step{r['observed_step']}"
            draw.text((x+2,y+241),label,fill='black')
        sheet.save(out/'cases'/f'contact_{si//12:02}.jpg',quality=92)
    (out/'cases.json').write_text(json.dumps(cases,indent=2));stats['prevalidated_projection_equivalence_checks']=projection_checks;stats['seconds']=time.monotonic()-start;stats['total_regions']=len(regions);stats['total_target_queries']=len(samples);stats['cases']=len(cases);(out/'stats.json').write_text(json.dumps(stats,indent=2));print(json.dumps(stats),flush=True)
if __name__=='__main__':main()
