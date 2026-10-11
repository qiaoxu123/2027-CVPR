"""Frozen official SigLIP2 variable templates; reuse image tower cache entirely."""
import json,time
import numpy as np,torch
from pathlib import Path
from graphnav.appearance_matching import ObservedSigLIPMatchingEncoder
from experiments.language_object_graph.run import load
from experiments.language_object_graph.objects import ObjectGraph
from experiments.semantic_polygon_grounding.audit import save,sha
from .resources import OUT
from .language import parse
from .visual import template

def main():
    torch.set_num_threads(1);start=time.time();dest=OUT/'frozen_text';dest.mkdir(exist_ok=True)
    if (dest/'complete.json').exists():raise RuntimeError('preserve frozen template cache')
    graphs={};texts={'an object'}
    for split in ('train','dev'):
        states,_,_=load(split)
        for state in states:
            m=state['metadata'];name=m['map_name']
            if name not in graphs:graphs[name]=ObjectGraph.restore(torch.load(f'runs/language_object_graph/graphs/{name}.pt',weights_only=True))
            texts.update(template(e) for e in parse(m['instruction'],graphs[name])['variables'])
    encoder=ObservedSigLIPMatchingEncoder(device='cuda');revision=encoder.model.config._commit_hash
    if revision!='1b426889ea62b5a72bf9839009a1b184bfc9c178':raise ValueError('SigLIP2 revision differs from reused vision tower')
    assert all(not p.requires_grad for p in encoder.model.parameters());result={}
    ordered=sorted(texts)
    for i in range(0,len(ordered),32):
        vectors=encoder.texts(ordered[i:i+32]).float().cpu();assert vectors.shape[1]==1024 and torch.isfinite(vectors).all()
        for t,v in zip(ordered[i:i+32],vectors):result[t]=v.half()
    torch.save(result,dest/'templates.pt');save(dest/'complete.json',dict(templates=len(result),sha=sha(dest/'templates.pt'),revision=revision,all_backbone_frozen=True,dimension=1024,finite=True,image_encoding_count=0,peak_gpu_bytes=torch.cuda.max_memory_allocated(),seconds=time.time()-start,source='public variable observables only, no private target identity'))
    print('TEXT COMPLETE',len(result),round(time.time()-start,1),flush=True)
if __name__=='__main__':main()
