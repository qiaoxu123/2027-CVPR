"""Evaluate frozen Dev-selected heads; never Seen/Unseen."""
import json,time,torch
from experiments.semantic_polygon_grounding.audit import save,sha
from experiments.bidirectional_core.stats import paired_difference
from .resources import OUT
from .train import overrides
from .evaluate import full_eval,observed_eval,dump,METRICS
from .diagnose import read_rows


def main():
    torch.set_num_threads(1);start=time.time();models={};manifest={}
    for name in ('geometry','rgb'):
        root=OUT/'training'/name;complete=json.loads((root/'complete.json').read_text());assert complete['epochs']==3 and complete['checkpoint_sha']==sha(root/'best.pt')
        ck=torch.load(root/'best.pt',weights_only=True,map_location='cpu');models[name]=ck['weights'];manifest[name]=complete
    with torch.no_grad():
        observed_eval(models)
        summary={}
        for split in ('train','dev'):
            base=read_rows(OUT/f'full_{split}_A0.jsonl.gz');summary[split]={}
            for name in ('geometry','rgb'):
                rows,metrics=full_eval(models[name].cuda(),methods=('LearnedGeom',),split=split,device='cuda',write=False,overrides=overrides(split,'real' if name=='rgb' else 'geometry'))
                rs=rows['LearnedGeom'];dump(OUT/f'full_{split}_Trained_{name}.jsonl.gz',rs);summary[split][name]=dict(metrics=metrics['LearnedGeom'],versus_A0={m:paired_difference(base,rs,m,seed=20261011) for m in METRICS})
            left=read_rows(OUT/f'full_{split}_Trained_geometry.jsonl.gz');right=read_rows(OUT/f'full_{split}_Trained_rgb.jsonl.gz');summary[split]['rgb_minus_geometry']={m:paired_difference(left,right,m,seed=20261011) for m in METRICS}
    save(OUT/'trained_full_metrics.json',dict(results=summary,seconds=time.time()-start,heads=manifest,no_seen=True,no_unseen=True,configuration='best full Internal Dev Goal R@1; ties earlier epoch, no family-subset model selection'))
if __name__=='__main__':main()
