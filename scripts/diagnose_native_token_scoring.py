"""Train-only scoring probe: full instruction versus prefix token evidence, fixed queries."""
import argparse,json
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from graphnav.object_query_belief import ObjectQueryRanker
from graphnav.checkpointing import atomic_json,file_sha256
from scripts.train_object_query_belief import load,collate


def main():
    p=argparse.ArgumentParser();p.add_argument('--train',required=True);p.add_argument('--checkpoint',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    meta,rows=load(a.train)
    if meta['partition']!='train':raise ValueError('Train-only diagnostic')
    state=torch.load(a.checkpoint,map_location='cpu',weights_only=True)
    if state['config']['relation']:raise ValueError('Plain native scorer diagnostic required')
    model=ObjectQueryRanker();model.load_state_dict(state['model']);model.eval().requires_grad_(False)
    totals={};covered=0;count=0;special=0;late=0
    with torch.inference_mode():
        for b in DataLoader(rows,batch_size=32,collate_fn=collate):
            features=b['features'].float();text=b['text'].float();valid=b['valid'];mask=b['text_mask'];logits=features@text.transpose(1,2)
            raw=logits.masked_fill(~mask[:,None],-1e4).amax(-1)
            with torch.autocast('cpu',dtype=torch.bfloat16):trained,_=model(features,text,mask,b['geometry'].float(),valid)
            correction=trained.float()-raw
            token=torch.arange(text.shape[1])[None];length=mask.sum(-1)
            nonspecial=mask&(token>0)&(token<length[:,None]-1)
            masks={'full':mask,'no_special':nonspecial}
            for n in (4,8,16):masks[f'prefix{n}']=nonspecial&(token<=n)
            good=((b['query_world'].float()-b['target_world'][:,None,:2]).norm(dim=-1)<=20)&valid
            has=good.any(-1);covered+=int(has.sum());count+=len(features)
            best=raw.masked_fill(~valid,-1e4).argmax(-1)
            winner_tokens=logits[torch.arange(len(features)),best].masked_fill(~mask,-1e4).argmax(-1)
            legal=valid.any(-1);special+=int(((winner_tokens==0)|(winner_tokens==length-1))[legal].sum());late+=int((winner_tokens>=length.float()/2)[legal].sum())
            for name,subset in masks.items():
                candidate=logits.masked_fill(~subset[:,None],-1e4).amax(-1)
                for suffix,score in [('native',candidate),('trained_correction',candidate+correction)]:
                    index=score.masked_fill(~valid,-1e4).argmax(-1)
                    hit=good[torch.arange(len(features)),index]
                    key=name+'_'+suffix;totals[key]=totals.get(key,0)+int(hit.sum())
    result=dict(partition='train',states=count,covered_states=covered,native_top_token_special_states=special,native_top_token_latter_half_states=late,ranking={name:dict(hits=n,all_state_R1=n/count,covered_state_R1=n/covered) for name,n in totals.items()},train_manifest_sha=file_sha256(Path(a.train)/'manifest.json'),native_sha=file_sha256(a.checkpoint),protocol='Fixed geometry-selected15 query cache. Full-context text encoding is unchanged. Prefixes count BERT wordpieces after CLS, not parsed target noun phrases. Original trained correction reused without fitting. GT scores ranking after every score is computed. Exploratory Train-only evidence; does not alter frozen fullSeen candidate or prove navigation gains.')
    atomic_json(result,Path(a.output));print(json.dumps(result,indent=2))


if __name__=='__main__':main()
