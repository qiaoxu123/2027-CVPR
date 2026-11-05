"""Frozen TRAIN-dev audit: current checkpoints, language role effects, RGB controls."""
import argparse
from collections import Counter
import json
from pathlib import Path
import time
import numpy as np
import torch
from graphnav.direct_target_data import load_model,batch_inputs
from graphnav.checkpointing import file_sha256,atomic_json
from scripts.train_direct_target import sample_rows,evaluate,coordinates


def main():
    p=argparse.ArgumentParser()
    for k in ('inventory','cache','checkpoint','original-v2','output'):p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);torch.set_num_threads(4)
    tasks=json.loads((a.inventory/'public_tasks.json').read_text());labels=json.loads((a.inventory/'private_labels.json').read_text())
    rows=sample_rows([t for t in tasks if t['subset']=='dev'],labels,a.cache)
    model=load_model(torch.load(a.checkpoint,map_location='cpu',weights_only=True)).eval().requires_grad_(False)
    metrics,predictions=evaluate(model,rows)
    original=load_model(torch.load(a.original_v2,map_location='cpu',weights_only=True)).eval().requires_grad_(False)
    baseline=[];control={'zero':[],'shuffled':[]};timing=[]
    shuffled=[s['observed']['image_features'] for s in rows if s['observed'] is not None]
    cursor=0
    with torch.inference_mode():
        for start in range(0,len(rows),8):
            samples=rows[start:start+8];inputs=batch_inputs(samples);dist=coordinates(samples,inputs)
            model.reset_episode()
            with torch.autocast('cuda',dtype=torch.bfloat16):
                field=original._geometry_field(inputs['static_inputs'],inputs['tokens'],inputs['token_mask'],inputs['node_emb'])
            original.reset_episode()  # no cross-batch cache, though inputs differ
            baseline.extend(float(dist[i,field.belief_logits[i].argmax()]) for i in range(len(samples)))
            for mode in control:
                altered=dict(inputs)
                if inputs['observed'] is not None:
                    obs=dict(inputs['observed']);image=obs['image_features'].clone()
                    if mode=='zero':image.zero_()
                    else:
                        for i,s in enumerate(samples):
                            if s['observed'] is not None:
                                image[i]=shuffled[(cursor+i+len(shuffled)//2)%len(shuffled)].float().cuda()
                    obs['image_features']=image;altered['observed']=obs
                torch.cuda.synchronize();t=time.perf_counter()
                with torch.autocast('cuda',dtype=torch.bfloat16):out=model.forward_direct_target(**altered)
                torch.cuda.synchronize();timing.append((time.perf_counter()-t)/len(samples)*1000)
                control[mode].extend(float(dist[i,out['field'].belief_logits[i].argmax()]) for i in range(len(samples)))
            cursor+=len(samples)
    # One query contributes equally even if fewer real RGB views exist. Each
    # state is reported as a state, never as a new independent target instance.
    summary=dict(state_metrics=metrics,original_geometry=dict(R20=float(np.mean(np.array(baseline)<=20)),NE=float(np.mean(baseline))),
        RGB_controls={k:dict(R20=float(np.mean(np.array(v)<=20)),NE=float(np.mean(v))) for k,v in control.items()},
        queries=len({r['task']['index'] for r in rows}),states=len(rows),
        actual_RGB_states=sum(r['observed'] is not None for r in rows),
        head_inference_ms_per_state=float(np.mean(timing[2:])),
        foundation_encoder_cost_included=False,checkpoint_sha256=file_sha256(a.checkpoint),
        evaluation_scope='TRAIN-map-held-out for new fitting; historical Geometry already pretrained on TRAIN')
    atomic_json(summary,a.output/'summary.json');atomic_json(predictions,a.output/'predictions.json')
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
