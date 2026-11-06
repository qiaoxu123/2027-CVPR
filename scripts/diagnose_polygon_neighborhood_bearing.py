"""Frozen TRAIN-only fit/holdout diagnosis, no retraining or navigation."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
import torch
from graphnav.token_language import FrozenBERTTokens
from graphnav.polygon_neighborhood_bearing import NeighborhoodBearingHead
from graphnav.checkpointing import atomic_json,file_sha256
from scripts.train_polygon_neighborhood_bearing import BearingExamples,evaluate


class InstructionShuffle:
    def __init__(self,base,indices):
        self.base=base;self.records=base.records
        order=np.random.default_rng(42).permutation(indices).tolist()
        self.swapped={i:base.records[order[(j+1)%len(order)]]['instruction'] for j,i in enumerate(order)}
    def __getitem__(self,i):
        item=dict(self.base[i]);item['text']=self.swapped.get(i,item['text']);return item
    def __len__(self):return len(self.base)


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--public-objects',type=Path,required=True);a=p.parse_args();torch.set_num_threads(4)
    conf=json.loads((a.run/'protocol.json').read_text());data=json.loads((a.run/'data_manifest.json').read_text());refs=max(len(r['landmarks']) for r in data['records']);assert refs<=16
    ds=BearingExamples(data['records'],json.loads(a.public_objects.read_text()),max_refs=16,neighbors=conf['neighbors'],radius_m=conf['radius_m'])
    bert=FrozenBERTTokens(conf['BERT_name'],device='cuda',max_length=conf['BERT_max_tokens']);bert.load_cache(a.run/'bert_cache.pt')
    # Validate old-cache provenance against actual fresh pretrained forward.
    audits=[]
    for i in [data['fit_indices'][0],data['heldout_indices'][0],data['fit_indices'][-1],data['heldout_indices'][-1]]:
        text=ds.records[i]['instruction'];cached=bert.cache.pop(text);bert._load([text]);fresh=bert.cache[text]
        delta=float((cached[0]-fresh[0]).abs().max());audits.append(dict(index=i,max_abs_token_delta=delta,token_count=len(fresh[0])));assert torch.allclose(cached[0],fresh[0],atol=2e-5,rtol=2e-5),'Frozen BERT cache mismatches actual model'
    initial=[];result={};fit_hist=np.zeros(8,dtype=int)
    for i in data['fit_indices']:
        y=ds[i]['labels'];fit_hist+=np.bincount(y[y>=0],minlength=8)
    result['fit_label_histogram']=fit_hist.tolist();result['BERT_cache_vs_actual_forward']=audits
    for arm,geom in [('text_name_control',False),('public_polygon_neighborhood',True)]:
        state=torch.load(a.run/(arm+'_last.pt'),weights_only=True,map_location='cpu');initial.append(state['initial_digest']);assert state['epoch']==conf['epochs'] and state['trained_on_heldout'] is False
        model=NeighborhoodBearingHead().cuda().eval().requires_grad_(False);model.load_state_dict(state['model'],strict=True)
        fit,fitrows=evaluate(model,ds,data['fit_indices'],bert,'cuda',geometry_enabled=geom,batch_size=conf['batch'])
        result[arm+'_fit']=fit
        with (a.run/(arm+'_fit_rows.jsonl')).open('w') as f:
            for row in fitrows:f.write(json.dumps(row)+'\n')
        if geom:
            shuffled=InstructionShuffle(ds,data['heldout_indices'])
            report,rows=evaluate(model,shuffled,data['heldout_indices'],bert,'cuda',geometry_enabled=True,batch_size=conf['batch'])
            result['B_shuffled_instruction']=report;result['shuffle_changed_queries']=sum(shuffled.swapped[i]!=ds.records[i]['instruction'] for i in data['heldout_indices'])
            with (a.run/'B_shuffled_rows.jsonl').open('w') as f:
                for row in rows:f.write(json.dumps(row)+'\n')
        del model;torch.cuda.empty_cache()
    assert len(set(initial))==1;result['identical_AB_initial_digest']=initial[0];result['catalog_SHA256']=file_sha256(a.public_objects)
    atomic_json(result,a.run/'diagnostics.json');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
