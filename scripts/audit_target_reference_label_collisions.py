"""Private TRAIN loss audit: world-radius positives versus instance proxies.

Produces diagnostics and uncertain loss labels only; no inference entry is
modified. World-near points are never called verified instance matches.
"""
import argparse
from collections import Counter,defaultdict
import json
from pathlib import Path
import numpy as np
import shapely
from shapely.geometry import Polygon
import torch
from graphnav.checkpointing import atomic_json,atomic_save,file_sha256
from graphnav.public_reference_supervision import patch_world_windows,reference_region_labels
from graphnav.training_instance_labels import instance_core_labels
from scripts.analyze_instruction_landmark_association import PublicNames,iter_train


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('root','catalog','annotations','output'):p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args();r=a.root
    if a.output.exists():raise FileExistsError(a.output)
    catalog=json.loads(a.catalog.read_text());names=PublicNames(catalog)
    partition=json.loads((r/'chain_train400_features/manifest.json').read_text())['map_assignment']
    registered=defaultdict(set)
    for _,raw in iter_train(a.annotations):
        mp=f"{raw['area']}_block_{raw['block']}"
        if partition.get(mp)!='fit':continue
        for oid in raw['object_ids']:
            if catalog[mp][str(oid)]['object_type']=='Car':registered[mp].add(str(oid))
    car_registry={m:(sorted(ids),np.asarray([catalog[m][i]['position'][:2] for i in sorted(ids)],float)) for m,ids in registered.items()}
    source=[('old900','chain_train400_features','fixed_yaw_train400_s1','public_deployment_filter.json'),
            ('new774','scoped_between_train294_features_v1','scoped_between_train294_observations_v1','scoped_between_train294_public_filter_v1.json')]
    rows=[];loss_labels={};all_counts=defaultdict(Counter)
    for tag,cache,observations,filter_name in source:
        allowed=set(json.loads((r/filter_name).read_text())['frame_keys'])
        frames={x['frame_key']:x for x in map(json.loads,(r/observations/'public_frames.jsonl').read_text().splitlines()) if x['frame_key'] in allowed}
        truth={x['frame_key']:x for x in map(json.loads,(r/observations/'labels_private.jsonl').read_text().splitlines()) if x.get('frame_key') in frames}
        manifest=json.loads((r/cache/'manifest.json').read_text());assert manifest['complete']
        for chunk in manifest['chunks']:
            path=r/cache/chunk['file'];assert file_sha256(path)==chunk['sha256']
            for f in torch.load(path,weights_only=True)['records']:
                key=f['frame_key']
                if f['subset']!='fit' or key not in allowed:continue
                label=truth[key];frame=frames[key];mp=f['map_name'];oid=str(label['physical_group'][1]);obj=catalog[mp][oid]
                world=f['candidate_world'][784:].numpy();valid=f['local_valid'].numpy();target=np.asarray(label['target_world'][:2])
                distance=np.linalg.norm(world-target,axis=-1);near=(distance<=20)&valid
                contour=np.asarray(obj['contour'],float);poly=Polygon(contour) if contour.ndim==2 and contour.shape[1]==2 and len(contour)>=3 else None
                mask=np.full(len(world),-1,np.int8);inside=np.zeros(len(world),bool);outside=np.zeros(len(world),bool)
                if poly is not None and poly.is_valid and poly.area>0:
                    points=shapely.points(world);inside=shapely.covers(poly,points)
                    outside=~shapely.covers(poly.buffer(2),points)
                    windows=patch_world_windows(frame['footprint_world'])[:,0]
                    support=valid&(f['quality'][:,0].numpy()>=.9)
                    # Same established boundary-uncertain region codec. Small
                    # objects may have zero positives; never force a substitute.
                    mask,_=reference_region_labels(windows,contour,support,margin_m=.5)
                named_inside=np.zeros(len(world),bool);named_other_ids=[]
                for rid in sorted({i for m in names.match(mp,frame['instruction']) for i in m['public_ids']}):
                    if rid==oid:continue  # annotation-side exclusion of target itself
                    c=np.asarray(catalog[mp][rid]['contour'],float)
                    if c.ndim!=2 or c.shape[1]!=2 or len(c)<3 or not np.isfinite(c).all():continue
                    pol=Polygon(c)
                    if not pol.is_valid or pol.area<=0:continue
                    named_inside|=shapely.covers(pol,shapely.points(world));named_other_ids.append(rid)
                core=np.full(len(world),-1,np.int8)
                if label['category']=='Car':
                    ids,xy=car_registry[mp];core=instance_core_labels(world,valid,xy,ids,oid)
                count=dict(frames=1,valid_candidates=int(valid.sum()),coarse20_positive=int(near.sum()),
                    coarse20_inside_target_contour=int((near&inside).sum()),
                    coarse20_outside_target_buffer2=int((near&outside).sum()),
                    coarse20_inside_other_named_region=int((near&named_inside).sum()),
                    coarse20_target_car_core=int((near&(core==1)).sum()),
                    coarse20_other_car_core=int((near&(core==0)).sum()),
                    frames_with_coarse20=int(near.any()),frames_with_core_positive=int(((core==1)&valid).any()),
                    frames_with_patch_positive=int((mask==1).any()),frames_with_patch_pos_and_neg=int((mask==1).any() and (mask==0).any()))
                for group in (tag+'::'+label['category'],'all::'+label['category']):all_counts[group].update(count)
                rows.append(dict(frame_key=key,task_key=f['task_key'],category=label['category'],source=tag,
                    counts=count,named_region_candidates=len(named_other_ids),
                    catalog_position_error_m=float(np.linalg.norm(target-np.asarray(obj['position'][:2]))),
                    visibility='unknown',pixel_gt=False))
                loss_labels[key]=dict(target_patch_geometry=torch.from_numpy(mask),registered_car_core=torch.from_numpy(core))
    a.output.mkdir(parents=True);atomic_save(loss_labels,a.output/'loss_labels_private.pt');atomic_json(rows,a.output/'records.json')
    summary=dict(frames=len(rows),targets=len({x['task_key'] for x in rows}),counts={k:dict(v) for k,v in all_counts.items()},
        catalog_position_max_error_m=max(x['catalog_position_error_m'] for x in rows),
        inference_modified=False,normal_forward_reads_private_labels=False,
        weak_label_definition='1-patch90% inner coverage,0.5m inner/outer uncertainty margin,outer overlap<=1% negative; separate collision statistic uses target buffer2m; car registered cores radius2m,neighbor margin0.75m',
        note='Geometry proxies only. Named-region collisions are candidate polygons, not verified semantic reference roles. Categories overlap; never add collision counts as exclusive classes.',
        sha256={k:file_sha256(v) for k,v in dict(catalog=a.catalog,annotations=a.annotations,producer=Path(__file__),labels=a.output/'loss_labels_private.pt').items()})
    atomic_json(summary,a.output/'summary.json');print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':main()
