"""Post-forward geometry-predicate diagnostics by relation kind; no new classifier claim."""
import argparse,json
from pathlib import Path
from types import SimpleNamespace
from collections import defaultdict
import numpy as np
import torch
from torch.utils.data import DataLoader
from graphnav.candidate_geometry import FullCandidateDataset,collate_full
from graphnav.full_experiment import _batch_inputs,new_model
from graphnav.static_controls import graph_control
from graphnav.relation_supervision import legal_region
from graphnav.token_language import FrozenBERTTokens
from graphnav.checkpointing import file_sha256,provenance


def main():
    p=argparse.ArgumentParser();p.add_argument('--records',required=True);p.add_argument('--bert-cache',required=True);p.add_argument('--checkpoints',nargs='+',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();out=Path(a.output)
    if out.exists():raise ValueError('refusing predicate overwrite')
    dataset=FullCandidateDataset(a.records,max_roads=0)
    loader=DataLoader(dataset,batch_size=32,collate_fn=collate_full,num_workers=0)
    encoder=FrozenBERTTokens('bert-base-uncased',device='cuda',max_length=96);encoder.load_cache(a.bert_cache)
    report={'source_sha256':file_sha256(a.records),'provenance':provenance(),'models':{},'definition':'Post-forward Top1 geometric predicate agreement with actual target predicate. Not a trained language relation classifier; thresholds are auxiliary supervision conventions. Positive-target satisfaction and contradictory source relations reported separately; no GT enters inputs.'}
    for entry in a.checkpoints:
        name,path=entry.split('=',1);ck=torch.load(path,map_location='cpu',weights_only=True);model=new_model(SimpleNamespace(**ck['args']),torch.device('cuda')).eval();model.load_state_dict(ck['model'],strict=True)
        variant=ck.get('variant','geometry');control=ck['args'].get('graph_control','full');counts=defaultdict(lambda:defaultdict(int));first=0
        with torch.inference_mode():
            for batch in loader:
                inputs,tokens,mask,names=_batch_inputs(batch,encoder,torch.device('cuda'),variant);inputs=graph_control(inputs,control,batch['node_names'])
                with torch.autocast('cuda',dtype=torch.bfloat16):field=model(inputs,variant=variant,tokens=tokens,token_mask=mask,node_emb=names)
                winner=field.belief_logits.argmax(-1).cpu().numpy()
                for j,cell in enumerate(winner):
                    row=dataset.records[first+j];graph=dataset._prepared(first+j)[0]
                    low=np.array(row['world_bounds'][:2]);span=np.array(row['world_bounds'][2:])-low
                    # Read private target only AFTER forward.
                    points=low+np.stack([dataset.grid_xy[cell],row['target_xy']])*span
                    for relation in graph['relations']:
                        result=legal_region(points,graph['entities'],relation)
                        if result is None:counts[relation.kind]['unreliable_or_unsupported']+=1;continue
                        c=counts[relation.kind];c['relations']+=1;c['predicate_agreement']+=int(result[0]==result[1]);c['target_satisfies']+=int(result[1]);c['prediction_satisfies']+=int(result[0]);c['positive_target_correct']+=int(result[0] and result[1])
                        c['TP' if result[0] and result[1] else 'FP' if result[0] else 'FN' if result[1] else 'TN']+=1
                first+=len(winner)
        result={}
        for kind,c in counts.items():
            result[kind]=dict(c)
            if c['relations']:result[kind]['predicate_accuracy']=c['predicate_agreement']/c['relations'];result[kind]['target_satisfaction_fraction']=c['target_satisfies']/c['relations']
            if c['target_satisfies']:result[kind]['positive_target_satisfaction_recall']=c['positive_target_correct']/c['target_satisfies']
            result[kind]['precision']=c['TP']/max(c['TP']+c['FP'],1)
            result[kind]['recall']=c['TP']/max(c['TP']+c['FN'],1)
            result[kind]['F1']=2*c['TP']/max(2*c['TP']+c['FP']+c['FN'],1)
        report['models'][name]={'checkpoint_sha256':file_sha256(path),'variant':variant,'control':control,'by_kind':result}
        out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(report,indent=2));print(name+' complete',flush=True);del model

if __name__=='__main__':main()
