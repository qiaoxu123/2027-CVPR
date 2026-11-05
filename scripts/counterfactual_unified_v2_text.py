"""TRAIN public-Polygon direction flips, with explicit manual-rule interventions.

No GT coordinates, target IDs or true geographic north labels used. Existing
engine convention north=+Y is only an engineering response test (CRS unverified).
"""
import argparse,json,random,copy
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
from torch.utils.data import DataLoader
from graphnav.candidate_geometry import FullCandidateDataset,collate_full
from graphnav.full_relation_training import load_full_relation_model,full_relation_inputs
from graphnav.token_language import FrozenBERTTokens
from graphnav.unified_text_ablation import forward_rule_control
from graphnav.checkpointing import atomic_json,atomic_save,file_sha256


def main():
    p=argparse.ArgumentParser()
    for k in ('prepared-dir','existing-full-checkpoint','checkpoint-landmark-relation','output'):p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--samples',type=int,default=64);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    source=[json.loads(s) for s in (a.prepared_dir/'full_text.jsonl').read_text().splitlines()];by_map=defaultdict(list);seen=set()
    for r in source:
        if len(r['landmarks'])!=1:continue
        lm=r['landmarks'][0];key=(r['map_name'],lm['id'])
        if key in seen:continue
        seen.add(key);by_map[r['map_name']].append(r)
    rng=random.Random(42)
    for candidates in by_map.values():rng.shuffle(candidates)
    chosen=[]
    while len(chosen)<a.samples and any(by_map.values()):
        for m in sorted(by_map):
            if by_map[m] and len(chosen)<a.samples:chosen.append(by_map[m].pop())
    assert chosen
    generated=[]
    for sample in chosen:
        for word in ('north','south','east','west'):
            r=copy.deepcopy(sample);r['instruction']=word+' of '+r['landmarks'][0]['name'];r['target_xy']=[0.,0.];generated.append(r)
    dataset_path=a.output/'synthetic_no_GT.jsonl';dataset_path.write_text(''.join(json.dumps(r)+'\n' for r in generated))
    checkpoints=dict(full_text=a.existing_full_checkpoint,landmark_relation=a.checkpoint_landmark_relation);results={};saved={}
    for arm,path in checkpoints.items():
        state=torch.load(path,map_location='cpu',weights_only=True);model=load_full_relation_model(state).eval().requires_grad_(False);cfg=state['config']['geometry_args'];bert=FrozenBERTTokens(cfg['text_backbone'],device='cuda',max_length=cfg['max_text_length'])
        cache=path.parent/'bert_cache.pt'
        if cache.exists():bert.load_cache(cache)
        ds=FullCandidateDataset(dataset_path,grid_size=28,max_landmarks=cfg['max_landmarks'],max_roads=cfg['max_roads'],max_relations=cfg['max_relations'],prior_sigma_m=cfg['prior_sigma_m'],rule_sigma_m=cfg['rule_sigma_m'],deduplicate_queries=False)
        for mode in ('full','additive_off','manual_off'):
            fields={'geometry':[],'relation':[]};grids=[]
            for batch in DataLoader(ds,batch_size=8,shuffle=False,collate_fn=collate_full,num_workers=0):
                inp=full_relation_inputs(batch,bert,'cuda')
                with torch.inference_mode():out=forward_rule_control(model,inp,mode)
                for key in fields:fields[key].append(out[key+'_field'].belief_logits.cpu())
                grids.append(batch['grid_xy'])
            fields={k:torch.cat(v) for k,v in fields.items()};grid=torch.cat(grids).numpy();summary={}
            for head,z in fields.items():
                prob=z.softmax(-1).numpy().reshape(len(chosen),4,784);centers=(prob[...,None]*grid.reshape(len(chosen),4,784,2)).sum(2);stats=[]
                for i,r in enumerate(chosen):
                    size=np.array(r['physical_map_sizes']);center=np.array(r['landmarks'][0]['contour']).mean(0);regionN=grid[i*4,:,1]>center[1];regionE=grid[i*4,:,0]>center[0]
                    ns=float((centers[i,0,1]-centers[i,1,1])*size[1]);ew=float((centers[i,2,0]-centers[i,3,0])*size[0]);nm=float((prob[i,0]-prob[i,1])[regionN].sum());em=float((prob[i,2]-prob[i,3])[regionE].sum())
                    stats.append(dict(map_name=r['map_name'],name=r['landmarks'][0]['name'],north_south_centroid_shift_m=ns,east_west_centroid_shift_m=ew,north_mass_response=nm,east_mass_response=em,NS_TV=float(abs(prob[i,0]-prob[i,1]).sum()/2),EW_TV=float(abs(prob[i,2]-prob[i,3]).sum()/2)))
                summary[head]=dict(samples=len(stats),NS_engine_direction_response=float(np.mean([x['north_south_centroid_shift_m']>1e-5 for x in stats])),EW_engine_direction_response=float(np.mean([x['east_west_centroid_shift_m']>1e-5 for x in stats])),NS_mass_response_rate=float(np.mean([x['north_mass_response']>1e-7 for x in stats])),EW_mass_response_rate=float(np.mean([x['east_mass_response']>1e-7 for x in stats])),mean_NS_shift_m=float(np.mean([x['north_south_centroid_shift_m'] for x in stats])),mean_EW_shift_m=float(np.mean([x['east_west_centroid_shift_m'] for x in stats])),per_sample=stats)
            results[arm+'/'+mode]=summary;saved[arm+'/'+mode]=fields
            print(json.dumps(dict(arm=arm,mode=mode,summary={k:{x:y for x,y in v.items() if x!='per_sample'} for k,v in summary.items()})),flush=True)
        del model,bert;torch.cuda.empty_cache()
    atomic_json(dict(methods=results,world_convention='Existing code north=+Y, east=+X; local raster has no CRS/tags, NOT geographic-north validation',GT_used=False,source_samples=[dict(map_name=r['map_name'],landmarks=r['landmarks'],world_bounds=r['world_bounds']) for r in chosen],checkpoints={k:file_sha256(v) for k,v in checkpoints.items()}),a.output/'counterfactual.json');atomic_save(saved,a.output/'fields.pt')

if __name__=='__main__':main()
