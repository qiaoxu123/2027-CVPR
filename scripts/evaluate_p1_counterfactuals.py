"""Controlled word swaps and coordinate translations, with labels kept outside forward."""
import argparse
from collections import Counter
import copy
import json
from pathlib import Path
import re

import numpy as np
import torch
from torch.utils.data import DataLoader

from graphnav.bidirectional import INVERSE
from graphnav.candidate_geometry import FullCandidateDataset, collate_full
from graphnav.full_experiment import _batch_inputs
from graphnav.full_localization import FullTokenGeometryLocalizer
from graphnav.static_localization import static_recall
from graphnav.token_language import FrozenBERTTokens


def write_rows(path, rows):
    path.write_text("".join(json.dumps(r)+"\n" for r in rows))


def predict(model, rows, path, encoder, variant="geometry", capacity=16):
    write_rows(path, rows)
    dataset = FullCandidateDataset(path, grid_size=28,max_landmarks=capacity,max_roads=0,max_relations=8)
    loader = DataLoader(dataset,batch_size=32,num_workers=4,collate_fn=collate_full)
    scores, xy, anchors = [], [], []
    totals = Counter()
    with torch.no_grad():
        for batch in loader:
            x,t,m,n = _batch_inputs(batch,encoder,torch.device("cuda"),variant)
            with torch.autocast("cuda",dtype=torch.bfloat16):
                field = model(x,variant=variant,tokens=t,token_mask=m,node_emb=n)
            logits=field.belief_logits.float()
            totals.update(static_recall(logits,batch["target_xy"].cuda(),batch["map_sizes"].cuda(),batch["grid_xy"].cuda()))
            scores.append(logits.cpu())
            expected = (logits.softmax(-1)[...,None]*x["grid_xy"]).sum(1)
            xy.append(expected.cpu());anchors.append(batch["nodes"][:,0,:2])
    size=totals['count']
    metric={"samples":size,"NE_m":totals['NE_sum_m']/size,
            **{f"R@{k}/20m":totals[f"R@{k}/20m_count"]/size for k in (1,4,8)}}
    return torch.cat(scores),torch.cat(xy),torch.cat(anchors),metric


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",required=True)
    p.add_argument("--synthetic",required=True)
    p.add_argument("--real-unseen",required=True)
    p.add_argument("--checkpoints",nargs="+",required=True)
    args=p.parse_args()
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    synth=[json.loads(x) for x in Path(args.synthetic).read_text().splitlines()]
    original,swapped=[],[]
    for row in synth:
        anchor=np.asarray(row['landmarks'][0]['contour']).mean(0)
        mirrored=2*anchor-np.asarray(row['target_xy'])
        if np.any(mirrored<.02) or np.any(mirrored>.98):continue
        new=copy.deepcopy(row)
        new['instruction']=re.sub(r'\b(east|west|north|south)\b',lambda m:INVERSE[m[0]],new['instruction'])
        new['relation_direction']=INVERSE[row['relation_direction']]
        new['target_xy']=mirrored.tolist()
        original.append(row);swapped.append(new)
    real=[json.loads(x) for x in Path(args.real_unseen).read_text().splitlines()]
    eligible=[]
    for row in real:
        points=np.asarray([p for lm in row['landmarks'] for p in lm['contour']] + [row['target_xy']])
        if len(row['landmarks']) and points.min()>.01 and points.max()<.95:eligible.append(row)
    shifted=copy.deepcopy(eligible)
    origin=copy.deepcopy(eligible)
    for row in shifted:
        row['target_xy']=(np.asarray(row['target_xy'])+1/28).tolist()
        for lm in row['landmarks']:lm['contour']=(np.asarray(lm['contour'])+1/28).tolist()
    for row in origin:
        row['world_bounds']=(np.asarray(row['world_bounds'])+np.tile([1000,2000],2)).tolist()
    encoder=FrozenBERTTokens(device='cuda',max_length=512)
    report={"direction_rows":len(original),"translated_real_unseen_queries":len(eligible),
            "word_swap_labels":"controlled synthetic cardinal inverse; mirrored label is evaluation only",
            "within_map_translation":"all supplied polygons and target +1/28 in normalized x/y; fixed grid",
            "world_origin_translation":"world bounds +1000m,+2000m; normalized polygons unchanged",
            "models":{}}
    dirs=torch.tensor([{'east':0,'west':1,'north':2,'south':3}[r['relation_direction']] for r in original])
    axis=dirs//2;sign=torch.where(dirs%2==0,1.,-1.)
    def side(x,a):return ((x-a).gather(1,axis[:,None]).squeeze(1)*sign)>0
    for entry in args.checkpoints:
        label,path=entry.split('=',1)
        model=FullTokenGeometryLocalizer(hidden=256,heads=8,relative_position=True).cuda().eval()
        model.load_state_dict(torch.load(path,map_location='cuda',weights_only=True)['model'])
        for name,variant in [(label,'geometry')]+([('rules','rules')] if label==args.checkpoints[0].split('=')[0] else []):
            old,xy,anchor,om=predict(model,original,out/'synthetic_original.jsonl',encoder,variant,1)
            new,newxy,_,nm=predict(model,swapped,out/'synthetic_swapped.jsonl',encoder,variant,1)
            a,_,_,am=predict(model,eligible,out/'real_original.jsonl',encoder,variant)
            b,_,_,bm=predict(model,shifted,out/'real_shifted.jsonl',encoder,variant)
            c,_,_,cm=predict(model,origin,out/'real_origin.jsonl',encoder,variant)
            av=a.reshape(-1,28,28)[:,:-1,:-1].flatten(1)
            bv=b.reshape(-1,28,28)[:,1:,1:].flatten(1)
            centered=(av-av.mean(1,keepdim=True))-(bv-bv.mean(1,keepdim=True))
            report['models'][name]={"original_direction_accuracy":float(side(xy,anchor).float().mean()),
                "swapped_direction_accuracy":float((~side(newxy,anchor)).float().mean()),
                "prediction_direction_flipped_fraction":float((side(xy,anchor)!=side(newxy,anchor)).float().mean()),
                "synthetic_original":om,"synthetic_word_swapped":nm,
                "real_original":am,"real_within_map_shift":bm,"real_world_origin_shift":cm,
                "aligned_interior_centered_logit_MAE":float(centered.abs().mean()),
                "world_origin_logit_MAE":float((a-c).abs().mean())}
            (out/'counterfactual_report.json').write_text(json.dumps(report,indent=2))
            print(json.dumps({'model':name,**report['models'][name]}),flush=True)


if __name__=='__main__':main()
