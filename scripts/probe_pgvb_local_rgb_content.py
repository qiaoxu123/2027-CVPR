"""Frozen content probe: re-encode constant RGB with identical real support/masks."""
import argparse,hashlib,json,math,time
from pathlib import Path
import cv2
import numpy as np
import torch
from graphnav.checkpointing import atomic_json,file_sha256
from graphnav.appearance_matching import ObservedSigLIPMatchingEncoder
from graphnav.pgvb_training import pgvb_batch
from graphnav.polygon_guided_visual import PolygonGuidedVisualBelief
from graphnav.landmark_visual_only import visual_only_world_belief
from graphnav.native_dense_regions import native_dense_regions,sparse_view
from scripts.run_native_grid_pilot import pool
from scripts.audit_visual_only_landmarks import prediction_index
from scripts.evaluate_pgvb import aggregate


def main():
    p=argparse.ArgumentParser()
    for k in ('data','training','output'):p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args();torch.set_num_threads(4)
    if a.output.exists():raise FileExistsError(a.output)
    proof=json.loads((a.data/'manifest.json').read_text());freeze=json.loads((a.training/'freeze.json').read_text())
    assert file_sha256(a.data/'cache.pt')==proof['cache_sha256'] and file_sha256(a.training/'best.pt')==freeze['best_sha256']
    data=torch.load(a.data/'cache.pt',weights_only=True,map_location='cpu');selection=[]
    for typ in ('Building','Road'):
        candidates=[i for i,r in enumerate(data['records']) if r['subset']=='dev' and r['legacy_common'] and r['type']==typ]
        candidates.sort(key=lambda i:hashlib.sha256(('content-probe0:'+data['records'][i]['query_key']+data['frames'][data['records'][i]['frame_id']]['frame']['image_sha256']).encode()).hexdigest())
        used={};chosen=[]
        while len(chosen)<32:
            options=[i for i in candidates if i not in chosen];options.sort(key=lambda i:(used.get(data['records'][i]['query_key'],0),candidates.index(i)))
            i=options[0];chosen.append(i);key=data['records'][i]['query_key'];used[key]=used.get(key,0)+1
        selection+=chosen
    a.output.mkdir(parents=True)
    sealed=[dict(index=i,query_key=data['records'][i]['query_key'],image_sha256=data['frames'][data['records'][i]['frame_id']]['frame']['image_sha256']) for i in selection]
    atomic_json(sealed,a.output/'selection_before_prediction.json');selection_sha=file_sha256(a.output/'selection_before_prediction.json')
    model=PolygonGuidedVisualBelief().cuda().eval().requires_grad_(False);state=torch.load(a.training/'best.pt',map_location='cpu',weights_only=True);model.load_state_dict(state['model'])
    encoder=ObservedSigLIPMatchingEncoder(proof['backbone']);rows=[];errors=[];started=time.perf_counter()
    for off in range(0,len(selection),4):
        ids=selection[off:off+4];batch,labels,rr=pgvb_batch(data,ids);images=[];regions=[]
        for r in rr:
            f=data['frames'][r['frame_id']];real=cv2.imread(f['image_path']);assert file_sha256(f['image_path'])==f['frame']['image_sha256']
            depth_path=Path(f['image_path']).parent/f['frame']['depth']
            if not depth_path.exists():depth_path=Path(f['image_path']).parent/Path(f['frame']['depth']).name
            with np.load(depth_path) as z:depth=z['depth_valid'].copy()
            regions.append(sparse_view(native_dense_regions(dict(bgr=real,depth_valid=depth,footprint_world=f['frame']['footprint_world']))))
            gray=np.where((real.max(-1)>0)[...,None],np.full_like(real,128),np.zeros_like(real))
            images.extend((real,gray))
        pixels=encoder.processor(images=[cv2.cvtColor(x,cv2.COLOR_BGR2RGB) for x in images],return_tensors='pt').to('cuda')
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):patch=encoder.model.vision_model(**pixels).last_hidden_state
        original=[];constant=[]
        for j,r in enumerate(rr):
            real_features=pool(encoder,patch[j*2],regions[j]['masks']);gray_features=pool(encoder,patch[j*2+1],regions[j]['masks'])
            errors.append(float((real_features.float()-data['frames'][r['frame_id']]['image_features'].float()).abs().mean()))
            original.append(real_features);constant.append(gray_features)
        uv=torch.stack([data['frames'][r['frame_id']]['local_uv'] for r in rr]).cuda()
        for control,features in [('real_reencoded',original),('constant_rgb_reencoded',constant)]:
            x=dict(batch);x['image_features']=torch.stack(features).cuda()
            with torch.inference_mode():s=model(**x);field=visual_only_world_belief(local_uv=uv,scores=s,valid=x['valid'])
            for j,(i,r) in enumerate(zip(ids,rr)):
                f=data['frames'][r['frame_id']];prob=field['probabilities'][j].cpu().numpy();known=field['covered'][j].cpu().numpy();world=f['global_world'].numpy()
                k=prediction_index(prob,known,world);ne=float(r['distance'][k]);scores=s[j].cpu().numpy();valid=f['valid'].numpy();local_prob=torch.from_numpy(scores).masked_fill(~f['valid'],float('-inf')).softmax(-1).numpy()
                local_world=np.asarray(f['bounds'])[:2]+f['local_uv'].numpy()*(np.asarray(f['bounds'])[2:]-np.asarray(f['bounds'])[:2]);top=prediction_index(local_prob,valid,local_world)
                top5=np.argsort(-np.where(valid,scores,-np.inf))[:5]
                rows.append(dict(index=i,control=control,type=r['type'],query_key=r['query_key'],map_name=f['map_name'],ne=ne,success=ne<=20,mass=float(prob@r['area_fraction'].numpy()),
                    proxy_top1_label=int(r['labels'][top]),proxy_top5=bool((r['labels'][top5]==1).any())))
        print(json.dumps(dict(completed=off+len(ids),total=len(selection))),flush=True)
    summary={c:{t:aggregate([r for r in rows if r['control']==c and (t=='all' or r['type']==t)]) for t in ('all','Building','Road')} for c in ('real_reencoded','constant_rgb_reencoded')}
    atomic_json(dict(summary=summary,real_cached_feature_MAE_max=max(errors),selection_sha256=selection_sha,checkpoint_sha256=freeze['best_sha256'],
        seconds=time.perf_counter()-started,selection_used_labels=False,parameter_updates=0,
        caveat='Post-result frozen mechanism diagnostic, not a new independent validation set. Constant RGB preserves the original nonblack support layout and original ROI/depth masks. It removes semantic texture/color, not all positional/context signals.'),a.output/'metrics.json')
    atomic_json(rows,a.output/'predictions.json');print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
