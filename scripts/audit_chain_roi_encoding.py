"""Blind frozen ROI encoding on TRAIN-fit states; GT scored only after predictions."""
import argparse,hashlib,json,time
from pathlib import Path
from collections import defaultdict
import cv2
import numpy as np
import torch
from torch.nn import functional as F
from graphnav.appearance_matching import ObservedSigLIPMatchingEncoder
from graphnav.observed_candidate_crops import observed_candidate_crops
from graphnav.checkpointing import file_sha256


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('observations','cache','labels','backbone','output'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--per-category',type=int,default=6)
    p.add_argument('--observation-filter',type=Path)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError('preserve prior experiment')
    labels={x['frame_key']:x for x in map(json.loads,a.labels.read_text().splitlines())}
    public={x['frame_key']:x for x in map(json.loads,(a.observations/'public_frames.jsonl').read_text().splitlines())}
    allowed=set(json.loads(a.observation_filter.read_text())['frame_keys']) if a.observation_filter else None
    tasks=defaultdict(list)
    for k,label in labels.items():
        if allowed is None or k in allowed:tasks[label['task_key']].append(k)
    chosen=[]
    for category in ('Car','Building','Ground','Parking'):
        eligible=[t for t,keys in tasks.items() if labels[keys[0]]['category']==category]
        eligible.sort(key=lambda t:hashlib.sha256(('roi-blind-20261104:'+t).encode()).hexdigest())
        for task in eligible[:a.per_category]:chosen.append(max(tasks[task],key=lambda k:public[k]['step']))
    a.output.mkdir(parents=True);selection=dict(frame_keys=chosen,observation_filter_sha256=file_sha256(a.observation_filter) if a.observation_filter else None,selection='TRAIN-fit target-group hash, six/category; latest available reference frame, no target-distance or visibility filtering',feature_manifest_sha256=file_sha256(a.cache/'manifest.json'))
    (a.output/'selection.json').write_text(json.dumps(selection,indent=2))
    selected={};meta=json.loads((a.cache/'manifest.json').read_text())
    for chunk in meta['chunks']:
        for row in torch.load(a.cache/chunk['file'],map_location='cpu',weights_only=True)['records']:
            if row['frame_key'] in chosen:selected[row['frame_key']]=row
    if set(selected)!=set(chosen):raise ValueError('missing selected cache frames')
    encoder=ObservedSigLIPMatchingEncoder(str(a.backbone),device='cuda',microbatch=16)
    gy,gx=np.meshgrid(np.arange(1,24,2),np.arange(1,24,2),indexing='ij');centers=(np.stack((gx.ravel(),gy.ravel()),-1)+.5)*16-.5
    predictions=[];features=[];network_input_shape=None
    torch.cuda.reset_peak_memory_stats()
    for key in chosen:
        row=selected[key];frame=public[key];path=a.observations/frame['image']
        if file_sha256(path)!=frame['image_sha256']:raise ValueError('RGB changed')
        rgb=cv2.imread(str(path));valid=row['local_valid'].numpy();crops,regions=observed_candidate_crops(rgb,centers,valid)
        if crops and network_input_shape is None:network_input_shape=list(encoder.processor(images=[cv2.cvtColor(crops[0],cv2.COLOR_BGR2RGB)],return_tensors='pt')['pixel_values'].shape)
        torch.cuda.synchronize();started=time.perf_counter();vectors=encoder.images(crops).cpu();torch.cuda.synchronize();seconds=time.perf_counter()-started
        text=F.normalize(row['target_text'].float(),dim=-1);values={};rankings={}
        native=F.normalize(row['image_features'].float(),dim=-1)
        for scale in range(3):values[f'native_pool_{[1,3,7][scale]}']=(native[:,scale]*text).sum(-1).masked_fill(~row['local_valid'],float('-inf'))
        crop_features=torch.zeros(144,2,1024)
        for vector,region in zip(vectors,regions):crop_features[region['candidate'],region['scale']]=vector
        for scale,width in enumerate((48,112)):values[f'crop_{width}']=(crop_features[:,scale]*text).sum(-1).masked_fill(~row['local_valid'],float('-inf'))
        for name,score in values.items():rankings[name]=score.argsort(descending=True)[:min(5,int(valid.sum()))].tolist()
        predictions.append(dict(frame_key=key,rankings=rankings,crop_seconds=seconds,crops=len(crops),image_sha256=frame['image_sha256'],regions=regions))
        features.append(dict(frame_key=key,image_features=crop_features.half()))
        print(json.dumps(dict(frame=key,crops=len(crops),seconds=seconds)),flush=True)
    # Seal normal inference outputs before accessing target world coordinates for scoring.
    (a.output/'blind_predictions.json').write_text(json.dumps(predictions,indent=2));torch.save(features,a.output/'crop_features.pt')
    evaluated=[]
    for pred in predictions:
        key=pred['frame_key'];row=selected[key];label=labels[key];world=row['candidate_world'][784:];target=torch.tensor(label['target_world'][:2]);distance=(world-target).norm(dim=-1);valid=row['local_valid']
        result=dict(frame_key=key,task_key=row['task_key'],category=label['category'],nearest_world_ne_m=float(distance[valid].min()) if valid.any() else None,methods={})
        for method,ids in pred['rankings'].items():result['methods'][method]=dict(top1_ne_m=float(distance[ids[0]]) if ids else None,top5_ne_m=float(distance[ids].min()) if ids else None)
        evaluated.append(result)
    summary=dict(network_input_shape=network_input_shape,backbone_snapshot=str(a.backbone),physical_targets=len(chosen),frames=len(chosen),gpu_peak_gib=torch.cuda.max_memory_allocated()/2**30,crop_seconds=sum(x['crop_seconds'] for x in predictions),crops=sum(x['crops'] for x in predictions),
                 warning='Fit-set late reference states; world point distance proxy, NOT pixel-instance match or closed-loop SR. 48/112px crops are resized by frozen SigLIP to384, no new native detail.',
                 blind_predictions_sha256=file_sha256(a.output/'blind_predictions.json'),rows=evaluated)
    (a.output/'evaluation.json').write_text(json.dumps(summary,indent=2));print(json.dumps({k:v for k,v in summary.items() if k!='rows'}))


if __name__=='__main__':main()
