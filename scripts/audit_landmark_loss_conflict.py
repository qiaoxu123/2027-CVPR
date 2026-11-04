"""TRAIN-fit gradient diagnosis only, no optimizer or model update."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import random
import numpy as np
import torch
from graphnav.geometry_guided_data import load_landmark_data,batch_landmarks
from graphnav.geometry_guided_visual import LandmarkVisualBelief
from graphnav.world_aligned_landmark_loss import world_landmark_visual_loss
from graphnav.checkpointing import file_sha256


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('data','teacher','catalog','checkpoint','output'):p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args();torch.set_num_threads(4)
    rows=[r for r in load_landmark_data(a.data,a.teacher,a.catalog) if r['metadata']['subset']=='fit' and (r['labels']==1).any() and (r['labels']==0).any()]
    groups=defaultdict(list)
    for r in rows:groups[r['metadata']['query_key']].append(r)
    net=LandmarkVisualBelief().cuda();state=torch.load(a.checkpoint,weights_only=True,map_location='cpu');net.load_state_dict(state['model'])
    rng=random.Random(137);keys=sorted(groups);parameters=list(net.parameters());records=[]
    for i in range(16):
        selected=[]
        for key in rng.sample(keys,8):
            group=groups[key];selected+=rng.sample(group,2) if len(group)>=2 else [group[0],group[0]]
        batch=batch_landmarks(selected);loss=world_landmark_visual_loss(net,batch);vectors={}
        for component in ('region','distill','contrast'):
            gradients=torch.autograd.grad(loss[component],parameters,retain_graph=True,allow_unused=True)
            vectors[component]=torch.cat([(g if g is not None else torch.zeros_like(p)).flatten() for p,g in zip(parameters,gradients)])
        a1,b1=vectors['region'],vectors['distill']
        cosine=float(torch.nn.functional.cosine_similarity(a1[None],b1[None])[0])
        records.append(dict(batch=i,region_distill_cosine=cosine,region_gradient_norm=float(a1.norm()),distill_gradient_norm=float(b1.norm()),
            region_loss=float(loss['region'].detach()),distill_loss=float(loss['distill'].detach())))
    assert all(torch.equal(net.state_dict()[k].cpu(),v) for k,v in state['model'].items())
    result=dict(batches=16,fit_only=True,model_unchanged=True,checkpoint_sha256=file_sha256(a.checkpoint),
        negative_gradient_cosine_batches=sum(r['region_distill_cosine']<0 for r in records),
        median_gradient_cosine=float(np.median([r['region_distill_cosine'] for r in records])),rows=records,
        caution='Negative gradient cosine shows local objective conflict, not proof of a unique generalization or navigation failure cause.')
    a.output.write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2))


if __name__=='__main__':main()
