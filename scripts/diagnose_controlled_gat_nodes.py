"""Frozen Seen GAT interventions. These measure reliance, not retrained superiority."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader
from graphnav.candidate_geometry import FullCandidateDataset, collate_full
from graphnav.full_experiment import _batch_inputs, new_model
from graphnav.token_language import FrozenBERTTokens
from graphnav.static_localization import static_recall
from graphnav.static_controls import graph_control
from graphnav.semantic_geometry import EDGE_TYPES, NODE_TYPES
from graphnav.checkpointing import file_sha256, provenance


def main():
    p=argparse.ArgumentParser()
    for key in ('checkpoint','records','bert-cache','output'):p.add_argument('--'+key,required=True)
    a=p.parse_args();out=Path(a.output)
    if out.exists():raise ValueError('refusing diagnostic overwrite')
    state=torch.load(a.checkpoint,map_location='cpu',weights_only=True)
    cfg=SimpleNamespace(**state['args'])
    model=new_model(cfg,torch.device('cuda')).eval();model.load_state_dict(state['model'],strict=True)
    encoder=FrozenBERTTokens(cfg.text_backbone,device='cuda',max_length=96);encoder.load_cache(a.bert_cache)
    dataset=FullCandidateDataset(a.records,max_roads=0)
    loader=DataLoader(dataset,batch_size=32,num_workers=0,collate_fn=collate_full)
    arms=('full','self','zero-edge-geometry','no-relation-links','no-physical-links',
          'no-relation-nodes','zero-node-geometry','zero-node-names','reverse-edge-direction')
    totals={arm:{'count':0,'NE_sum_m':0.,**{f'R@{k}/20m_count':0 for k in (1,4,8)}} for arm in arms}
    report=dict(checkpoint_sha256=file_sha256(a.checkpoint),input_sha256=file_sha256(a.records),
                provenance=provenance(),definition='Post-hoc frozen-weight reliance on public graph inputs; rules, candidate pair geometry and full BERT words held fixed. Not independently retrained variants or proof of causal training gains.')
    with torch.inference_mode():
        for batch in loader:
            x,t,m,n=_batch_inputs(batch,encoder,torch.device('cuda'),'gat')
            links=(x['edges'][...,8+EDGE_TYPES['relation_anchor']]>0)|(x['edges'][...,8+EDGE_TYPES['anchor_relation']]>0)
            eye=torch.eye(x['adjacency'].shape[-1],device='cuda',dtype=torch.bool)[None]
            for arm in arms:
                y=dict(x);names=n
                if arm=='self':y=graph_control(x,'self')
                elif arm=='zero-edge-geometry':y=graph_control(x,'zero-geometry')
                elif arm=='no-relation-links':y['adjacency']=x['adjacency'] & ~links
                elif arm=='no-physical-links':y['adjacency']=x['adjacency'] & (eye|links)
                elif arm=='no-relation-nodes':
                    valid=x['node_valid'] & (x['node_types']!=NODE_TYPES['relation'])
                    y['node_valid']=valid
                    y['adjacency']=x['adjacency'] & valid[:,:,None] & valid[:,None,:]
                elif arm=='zero-node-geometry':y['nodes']=torch.zeros_like(x['nodes'])
                elif arm=='zero-node-names':names=torch.zeros_like(n)
                elif arm=='reverse-edge-direction':
                    y['edges']=x['edges'].clone();y['edges'][...,[0,1,4,5]]*=-1
                with torch.autocast('cuda',dtype=torch.bfloat16):field=model(y,variant='gat',tokens=t,token_mask=m,node_emb=names)
                metrics=static_recall(field.belief_logits.float(),batch['target_xy'].to('cuda'),batch['map_sizes'].to('cuda'),x['grid_xy'])
                for key,value in metrics.items():totals[arm][key]+=value
    report['arms']={arm:dict(samples=v['count'],NE_m=v['NE_sum_m']/v['count'],
        **{f'R@{k}/20m':v[f'R@{k}/20m_count']/v['count'] for k in (1,4,8)}) for arm,v in totals.items()}
    out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(report,indent=2));print(json.dumps(report['arms'],indent=2))


if __name__=='__main__':main()
