"""TRAIN fit full-observed region oracle; anonymous contours never enter policy."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import cv2
import numpy as np
import shapely
from shapely.geometry import Polygon
import torch
from graphnav.checkpointing import atomic_json,file_sha256
from graphnav.landmark_projection import world_to_frame
from graphnav.visual_chain_position import WorldLikelihoodPositionHead
from scripts.train_local_chain_position import records,collate,batch_inputs


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('root','data-root','sbf-source','requirements','sparse-result','output'):p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError('preserve previous diagnostic')
    torch.set_num_threads(4);r=a.root
    meta=json.loads((r/'chain_train400_features/manifest.json').read_text());maps={m for m,s in meta['map_assignment'].items() if s=='fit'}
    sys.path.insert(0,str(a.sbf_source));from sbfnav.dataset import CityNavDataset
    ds=CityNavDataset(a.data_root,'train_seen',version='processed_citynav')
    if file_sha256(a.data_root/'cityrefer/objects.json')!=meta['public_catalog_sha256']:raise ValueError('catalog mismatch')
    objects=json.loads((a.data_root/'cityrefer/objects.json').read_text());registry={m:set() for m in maps}
    for raw in ds._raw_records:
        name=f"{raw['area']}_block_{raw['block']}"
        if name in maps:registry[name].add(str(raw['object_ids'][0]))
    classes=('Building','Parking','Ground');regions={}
    for name,ids in registry.items():
        by_class={c:[] for c in classes}
        for oid in ids:
            obj=objects[name][oid];kind=obj['object_type']
            if kind in by_class:
                poly=Polygon(obj['contour'])
                if poly.is_valid and poly.area>0:by_class[kind].append(poly)
        regions[name]=[(shapely.union_all(by_class[c]).buffer(-2),shapely.union_all(by_class[c]).buffer(2)) for c in classes]
    allowed=set(json.loads((r/'public_deployment_filter.json').read_text())['frame_keys']);role_vectors=torch.load(r/'reference_vectors_v2.pt',map_location='cpu',weights_only=True)
    rows=records(r/'chain_train400_features',r/'fixed_yaw_train400_s1/labels_fit.jsonl','fit',role_vectors,'reference',allowed)
    requirements=json.loads(a.requirements.read_text());frames={x['frame_key']:x for x in map(json.loads,(r/'fixed_yaw_train400_s1/public_frames.jsonl').read_text().splitlines())}
    sparse=json.loads(a.sparse_result.read_text());baseline={x['frame_key']:x for x in sparse['rows']['FrozenV3']}
    model=WorldLikelihoodPositionHead().cuda().eval().requires_grad_(False);ck=r/'world_v3_envelope_s0/best.pt';model.load_state_dict(torch.load(ck,map_location='cpu',weights_only=True)['model'])
    cache={};stats=Counter();output=[]
    for start in range(0,len(rows),8):
        batch_rows=rows[start:start+8]
        with torch.inference_mode():logits=model(**batch_inputs(collate(batch_rows),'cuda'))[:,:784].cpu()
        for i,row in enumerate(batch_rows):
            key=row['frame_key'];name=row['map_name'];world=row['candidate_world'][:784].numpy();proposal=requirements[row['task_key']]['proposal'];choice=int(logits[i].argmax());mask=np.zeros(784,bool)
            assert choice==baseline[key]['chosen_index']
            if name not in cache:
                points=shapely.points(world);inside=np.stack([shapely.covers(inner,points) for inner,_ in regions[name]],-1);expanded=np.stack([shapely.covers(outer,points) for _,outer in regions[name]],-1)
                labels=np.full(784,-1,np.int8)
                for c in range(3):labels[inside[:,c]&~expanded[:,np.arange(3)!=c].any(-1)]=c
                cache[name]=(world.copy(),labels)
            saved,labels=cache[name];assert np.allclose(saved,world,atol=.001)
            frame=frames[key];image_path=r/'fixed_yaw_train400_s1'/frame['image'];depth_path=r/'fixed_yaw_train400_s1'/frame['depth']
            if file_sha256(image_path)!=frame['image_sha256'] or file_sha256(depth_path)!=frame['depth_sha256']:raise ValueError('observed RGB/depth changed')
            rgb=cv2.imread(str(image_path));depth=np.load(depth_path)['depth_valid'];pixels=world_to_frame(world,frame['footprint_world'],rgb.shape)
            observed=np.isfinite(pixels).all(-1)&(pixels[:,0]>=0)&(pixels[:,0]<=383)&(pixels[:,1]>=0)&(pixels[:,1]<=383)
            indices=np.rint(pixels[observed]).astype(int);support=np.zeros(784,bool);support[observed]=(rgb[indices[:,1],indices[:,0]].max(-1)>0)&depth[indices[:,1],indices[:,0]]
            truth=row['target_world'].numpy();old_ne=float(np.linalg.norm(world[choice]-truth));status='no_requirement';opportunity=False
            if proposal:
                expected=classes.index(proposal['region']);mask=support&(labels>=0)&(labels!=expected)
                opportunity=bool((support&(labels==expected)&(np.linalg.norm(world-truth,axis=-1)<=20)).any())
                status='outside_real_observed_support' if not support[choice] else 'observed_unknown_region' if labels[choice]<0 else 'observed_expected_region' if labels[choice]==expected else 'observed_incompatible_region'
            if old_ne>20:
                stats['baseline_failures']+=1;stats['failure_'+status]+=1;stats['failures_with_known_expected_20m_support']+=int(opportunity)
            if mask.all():mask[:]=False;stats['all_rejected_fallback']+=1
            new_choice=int(logits[i].masked_fill(torch.from_numpy(mask),float('-inf')).argmax());ne=float(np.linalg.norm(world[new_choice]-truth))
            output.append(dict(frame_key=key,task_key=row['task_key'],category=row['category'],proposal=proposal,
                baseline_ne_m=old_ne,baseline_success=old_ne<=20,ne_m=ne,success=ne<=20,
                old_peak_region_status=status,known_expected_20m_support=opportunity,rejected_cells=int(mask.sum())))
    summary=dict(frames=len(output),baseline_correct=sum(x['baseline_success'] for x in output),oracle_correct=sum(x['success'] for x in output),
        baseline_ne_m=float(np.mean([x['baseline_ne_m'] for x in output])),oracle_ne_m=float(np.mean([x['ne_m'] for x in output])),
        corrected=[x['frame_key'] for x in output if x['success'] and not x['baseline_success']],harmed=[x['frame_key'] for x in output if not x['success'] and x['baseline_success']],
        stats=dict(stats),warning='Full observed TRAIN contour-region Oracle only. No normal model receives anonymous contours. 20m same-class support is not instance matching or a proven recoverable count.')
    a.output.mkdir(parents=True);atomic_json(dict(summary=summary,rows=output,checkpoint_sha256=file_sha256(ck),requirements_sha256=file_sha256(a.requirements),source_sha256=file_sha256(__file__)),a.output/'results.json');print(json.dumps(summary))


if __name__=='__main__':main()
