"""Frozen foundation controls with the same Dev-derived residual gain.

No per-foundation Goal tuning on Seen: representations are those fixed by the
earlier internal Dev comparison; gain is the already-fixed nonzero diagnostic.
"""
import argparse
import json
from pathlib import Path
import time
import cv2
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from .compare_foundations import load_official
from .city_features import representations
from .evaluate_features import load
from .goal_evidence import predict,summary
from experiments.bidirectional_core.cache import CachedCoreDataset,DEFAULT_DATA
from experiments.bidirectional_core.stats import paired_difference
from graphnav.renderer import CityNavObservedViewRenderer
from graphnav.checkpointing import file_sha256


def run(args):
    root=Path(args.output)
    if root.exists():raise ValueError('refusing foundation Goal overwrite')
    root.mkdir(parents=True);started=time.time();cv2.setNumThreads(1)
    frozen=json.loads(Path(args.selection).read_text())
    if not frozen['frozen']:raise ValueError('unfrozen representation')
    method=frozen['models'][args.model]['method'];representation,language=method.rsplit('_',1)
    if representation not in ['tight','context']:raise ValueError('this fixed pilot controls support tight/context only')
    rows=load(args.features);dataset=CachedCoreDataset(args.cache,'E1')
    model,preprocess,tokenizer,quick=load_official(args.checkpoint,args.model);renderer=CityNavObservedViewRenderer(Path(args.data_root)/'rgbd',root/'renderer_private',image_size=384)
    raw=json.loads((Path(args.data_root)/'cityrefer/objects.json').read_text());records={k:{} for k in ['real','zero','shuffled']};encoded=[]
    for index,row in enumerate(rows):
        m=row['metadata'];p=row['public'];regions=m['roi_objects'];observation=renderer.current_rgbd(m['map_name'],np.asarray(m['actual_camera_pose'],np.float64),float(p['ground_level']))
        crops=[representations(observation['bgr'],r,dict(id=r['object_id'],contour=raw[m['map_name']][r['object_id']]['contour']),observation['footprint_world'])[0 if representation=='tight' else 2] for r in regions]
        donors=[]
        for j,region in enumerate(regions):
            peers=sorted((k for k,r in enumerate(regions) if r['object_type']==region['object_type']),key=lambda k:regions[k]['object_id']);donors.append(peers[(peers.index(j)+1)%len(peers)])
        controls=dict(real=crops,zero=[np.zeros_like(c) for c in crops],shuffled=[cv2.resize(crops[k],(c.shape[1],c.shape[0])) for c,k in zip(crops,donors)])
        text_input=tokenizer([m['instruction'] if language=='full' else row['phrase']]).cuda()
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):text=F.normalize(model.encode_text(text_input).float(),dim=-1)
        saved={}
        for kind,views in controls.items():
            images=torch.stack([preprocess(Image.fromarray(cv2.cvtColor(v,cv2.COLOR_BGR2RGB))) for v in views]).cuda()
            with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):visual=F.normalize(model.encode_image(images).float(),dim=-1)
            score=(visual@text.T).T.cpu()*20
            if not torch.isfinite(score).all():raise ValueError('nonfinite actual native comparison')
            records[kind][m['sample_id']]=dict(score=score,confidence=p['roi_valid'][None].float(),observed=p['roi_valid'][None],row=row);saved[kind]=score
        encoded.append(dict(sample_id=m['sample_id'],scores=saved,donor_indices=donors))
        if index%50==0:print(json.dumps(dict(states=index+1,total=len(rows),seconds=time.time()-started)),flush=True)
    predictions={'geometry_only':predict(dataset,{},0,0,False)}
    for kind,data in records.items():predictions[kind+'_rgb']=predict(dataset,data,.025,0,False)
    predictions['real_with_history']=predict(dataset,records['real'],.025,0,True)
    for name,data in predictions.items():
        with (root/(name+'_predictions.jsonl')).open('w') as h:
            for row in data:h.write(json.dumps(row)+'\n')
    paired={name:{key:paired_difference(predictions['geometry_only'],data,key,repetitions=2000) for key in ['r1','r4','r8','ne','forward_r1']} for name,data in predictions.items()}
    rgb={kind:{key:paired_difference(predictions[kind+'_rgb'],predictions['real_rgb'],key,repetitions=2000) for key in ['r1','forward_r1','weak_forward_r1']} for kind in ['zero','shuffled']}
    delta=paired['real_rgb']['r1'];gate=delta['map_cluster_95ci'][0]>0 and delta['mean']>0 and rgb['shuffled']['r1']['mean']>0
    result=dict(models={n:summary(data) for n,data in predictions.items()},paired_vs_geometry=paired,real_minus_control=rgb,closedloop_and_unseen_gate_passed=gate,actual_states=len(dataset),actual_feature_states=len(rows),method=method,activation='QuickGELU' if quick else 'GELU',checkpoint_sha256=file_sha256(args.checkpoint),frozen_selection_sha256=file_sha256(args.selection),gain=.025,threshold=0,foundation_goal_threshold_tuning_on_seen=False,comparison_role='supplemental parallel frozen-native diagnosis; not production selection',control_policy='same-class current-view donors resized to destination actual tight/context rectangle; destination bbox size, geometry, language, visibility fixed; zero RGB rectangle; no future or GT donor choice',seconds=time.time()-started,peak_gpu_bytes=torch.cuda.max_memory_allocated())
    torch.save(encoded,root/'native_scores.pt');(root/'metrics.json').write_text(json.dumps(result,indent=2));print(json.dumps(dict(gate=gate,real_minus_geometry=delta),indent=2))


def main():
    p=argparse.ArgumentParser();p.add_argument('--cache',required=True);p.add_argument('--features',required=True);p.add_argument('--selection',required=True);p.add_argument('--model',choices=['georsclip','skyclip'],required=True);p.add_argument('--checkpoint',required=True);p.add_argument('--output',required=True);p.add_argument('--data-root',default=DEFAULT_DATA);run(p.parse_args())


if __name__=='__main__':main()
