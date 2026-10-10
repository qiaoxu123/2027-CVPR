"""Separate BERT-word, parsed-graph, and full pipeline direction effects; no SR claim."""
import argparse,json
from pathlib import Path
from types import SimpleNamespace
import torch
from torch.utils.data import DataLoader
from graphnav.candidate_geometry import FullCandidateDataset,collate_full
from graphnav.full_experiment import _batch_inputs,new_model
from graphnav.static_controls import graph_control
from graphnav.token_language import FrozenBERTTokens
from graphnav.checkpointing import file_sha256,provenance


def main():
    p=argparse.ArgumentParser();p.add_argument('--original',required=True);p.add_argument('--swapped',required=True);p.add_argument('--checkpoints',nargs='+',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();out=Path(a.output)
    if out.exists():raise ValueError('refusing factorial overwrite')
    old=FullCandidateDataset(a.original,max_landmarks=1,max_roads=0,max_relations=8)
    new=FullCandidateDataset(a.swapped,max_landmarks=1,max_roads=0,max_relations=8)
    if len(old)!=len(new):raise ValueError('counterfactual length mismatch')
    encoder=FrozenBERTTokens('bert-base-uncased',device='cuda',max_length=96)
    # Cache publicly observable inputs once for all checkpoint comparisons.
    pairs=[]
    for b,c in zip(DataLoader(old,batch_size=32,collate_fn=collate_full),DataLoader(new,batch_size=32,collate_fn=collate_full)):
        x,t,m,n=_batch_inputs(b,encoder,torch.device('cuda'),'geometry')
        y,u,v,w=_batch_inputs(c,encoder,torch.device('cuda'),'geometry')
        pairs.append((x,t,m,n,y,u,v,w,b['node_names'],c['node_names']))
    axis=torch.tensor([{'east':0,'west':0,'north':1,'south':1}[r['relation_direction']] for r in old.records],device='cuda')
    sign=torch.tensor([1 if r['relation_direction'] in ('east','north') else -1 for r in old.records],device='cuda')
    anchor=torch.tensor([r.get('anchor_center_xy',torch.tensor(r['landmarks'][0]['contour']).mean(0).tolist()) for r in old.records],device='cuda')
    report={'samples':len(old),'provenance':provenance(),'input_sha256':{'original':file_sha256(a.original),'swapped':file_sha256(a.swapped)},'models':{},'note':'Controlled synthetic cardinal labels. Expected-probability position side, not real target Recall or SR. Word-only fixes original geometry/rules/names; graph-only fixes original BERT tokens and explicit rule features, changes parsed graph/name embeddings. Full swap changes entire public pipeline.'}
    for item in a.checkpoints:
        name,path=item.split('=',1);ck=torch.load(path,map_location='cpu',weights_only=True)
        cfg=SimpleNamespace(**ck['args']);model=new_model(cfg,torch.device('cuda')).eval();model.load_state_dict(ck['model'],strict=True)
        variant=ck.get('variant','geometry');control=ck['args'].get('graph_control','full');pred={k:[] for k in ('original','full_swap','word_only','graph_only')}
        with torch.inference_mode():
            for x,t,m,n,y,u,v,w,bnames,cnames in pairs:
                gx=graph_control(x,control,bnames);gy=graph_control(y,control,cnames)
                graph=dict(gy,rules=gx['rules'],relation_fields=gx['relation_fields'])
                recipes={'original':(gx,t,m,n),'full_swap':(gy,u,v,w),'word_only':(gx,u,v,n),'graph_only':(graph,t,m,w)}
                for label,(inp,tokens,mask,names) in recipes.items():
                    with torch.autocast('cuda',dtype=torch.bfloat16):field=model(inp,variant=variant,tokens=tokens,token_mask=mask,node_emb=names)
                    pred[label].append((field.belief_logits.float().softmax(-1)[...,None]*inp['grid_xy']).sum(1))
        sides={k:((torch.cat(v)-anchor).gather(1,axis[:,None]).squeeze(1)*sign)>0 for k,v in pred.items()}
        result={'checkpoint_sha256':file_sha256(path),'variant':variant,'control':control,'original_direction_accuracy':float(sides['original'].float().mean())}
        for label in ('full_swap','word_only','graph_only'):
            result[label]={'flipped_fraction':float((sides[label]!=sides['original']).float().mean()),'inverse_direction_accuracy':float((~sides[label]).float().mean()),'both_directions_correct':float((sides['original'] & ~sides[label]).float().mean())}
        report['models'][name]=result;out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(report,indent=2));print(json.dumps({name:result}),flush=True)
        del model

if __name__=='__main__':main()
