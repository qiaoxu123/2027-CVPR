"""TRAIN-fit positional proxy audit; never reports pixel-instance accuracy."""
import argparse,json
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.nn import functional as F
from scripts.train_local_chain_position import records,collate,batch_inputs
from graphnav.visual_chain_position import WorldLikelihoodPositionHead
from graphnav.local_chain_raster import rasterized_goal_logprob
from graphnav.checkpointing import file_sha256


def summarize(rows):
    result=dict(frames=len(rows),physical_targets=len({x['task_key'] for x in rows}))
    for radius in (5,10,20):
        supported=[x for x in rows if x['nearest_m']<=radius]
        result[f'support_{radius}m']=len(supported)
        for name in ('trained_visual','native_cosine'):
            for k in (1,5):
                result[f'{name}_top{k}_{radius}m']=sum(x[f'{name}_top{k}_ne_m']<=radius for x in rows)
                result[f'{name}_top{k}_conditional_{radius}m']=sum(x[f'{name}_top{k}_ne_m']<=radius for x in supported)/len(supported) if supported else None
    result.update(prior_correct=sum(x['prior_ne_m']<=20 for x in rows),final_correct=sum(x['goal_ne_m']<=20 for x in rows),
        corrections=sum(x['prior_ne_m']>20 and x['goal_ne_m']<=20 for x in rows),harms=sum(x['prior_ne_m']<=20 and x['goal_ne_m']>20 for x in rows),
        prior_ne_m=float(np.mean([x['prior_ne_m'] for x in rows])),goal_ne_m=float(np.mean([x['goal_ne_m'] for x in rows])))
    wrong=[x for x in rows if x['prior_ne_m']>20]
    result['prior_wrong_frames']=len(wrong)
    result['prior_wrong_with_support20']=sum(x['nearest_m']<=20 for x in wrong)
    result['prior_wrong_with_visual_top1_20']=sum(x['trained_visual_top1_ne_m']<=20 for x in wrong)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('cache','labels','role-vectors','checkpoint','output'):p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError('preserve previous audit')
    torch.set_num_threads(4)
    roles=torch.load(a.role_vectors,weights_only=True);data=records(a.cache,a.labels,'fit',roles,'reference')
    state=torch.load(a.checkpoint,map_location='cpu',weights_only=True)
    if state['architecture']!='world-likelihood-v3':raise ValueError('V3 checkpoint required')
    model=WorldLikelihoodPositionHead().cuda().eval().requires_grad_(False);model.load_state_dict(state['model'])
    output=[]
    with torch.no_grad():
        for batch in DataLoader(data,batch_size=16,collate_fn=collate):
            x=batch_inputs(batch,'cuda');joint,visual=model(**x,return_visual=True)
            logits=rasterized_goal_logprob(joint,x['local_uv'],x['local_valid'],global_cells=784)
            cosine=(F.normalize(x['image_features'][:,:,0].float(),dim=-1)*F.normalize(x['target_text'].float(),dim=-1)[:,None]).sum(-1)
            order={name:scores.masked_fill(~x['local_valid'],float('-inf')).argsort(-1,descending=True).cpu() for name,scores in [('trained_visual',visual),('native_cosine',cosine)]}
            for i in range(len(batch['task_key'])):
                world=batch['candidate_world'][i];target=batch['target_world'][i];dist=(world-target).norm(dim=-1);valid=batch['local_valid'][i]
                local=dist[784:];nearest=float(local[valid].min()) if valid.any() else None
                row=dict(frame_key=batch['frame_key'][i],task_key=batch['task_key'][i],category=batch['category'][i],nearest_m=nearest if nearest is not None else 1e30,
                    valid_local_count=int(valid.sum()),prior_ne_m=float(dist[batch['prior_logits'][i].argmax()]),goal_ne_m=float(dist[logits[i].argmax().cpu()]))
                for name,indices in order.items():
                    for k in (1,5):
                        ids=indices[i,:min(k,int(valid.sum()))]
                        row[f'{name}_top{k}_ne_m']=float(local[ids].min()) if len(ids) else 1e30
                output.append(row)
    groups=defaultdict(list)
    for row in output:groups[row['category']].append(row)
    result=dict(warning='TRAIN fit, positional world-distance proxies only, no human pixel-instance labels, no visibility assertion; not generalization',
        checkpoint_sha256=file_sha256(a.checkpoint),feature_manifest_sha256=file_sha256(a.cache/'manifest.json'),
        all=summarize(output),by_category={k:summarize(v) for k,v in groups.items()},rows=output)
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='rows'}))


if __name__=='__main__':main()
