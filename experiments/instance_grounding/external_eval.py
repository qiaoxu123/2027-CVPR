"""Compare external-only heads on fixed external Dev (selection, not Test)."""
import argparse,json
from pathlib import Path
import torch
from experiments.bidirectional_core.stats import summarize_rows,paired_difference
from .dataset import external_rows
from .evaluate import load_model,predict_scores,family_rows
from .data import write

def run(a):
    out=Path(a.output)
    if out.exists():raise ValueError('no overwrite')
    out.mkdir(parents=True);torch.set_num_threads(2);rows=external_rows(a.features);results={}
    if a.black:
        black=torch.load(a.black,weights_only=True)
        for r in rows:r['pixel_zero']=black
    heads=json.loads(Path(a.heads).read_text())
    for name,path in {n+(('_'+c) if a.controls else ''):(p,c) for n,p in heads.items() for c in (['real','zero','shuffled'] if a.controls else ['real'])}.items():
        path,control=path;model=load_model(path);preds=predict_scores(model,rows,control);records=[]
        for r,p in zip(rows,preds):
            lab=r['labels'];valid=r['public']['roi_valid']&lab['semantic_valid_mask'];positive=lab['positive_roi_mask']&valid
            if not positive.any() or not (valid&~positive).any():continue
            rank=p['score'].masked_fill(~valid,-torch.inf).argsort(descending=True);m=r['metadata'];records.append(dict(sample_id=m['sample_id'],family_id=m['family_id'],map_name=m['map_name'],metrics=dict(region_r1=float(positive[rank[0]])),score=p['score'].tolist(),match_probability=p['prob'][:,0].tolist()))
        results[name]=records;(out/(name+'_predictions.jsonl')).write_text(''.join(json.dumps(r)+'\n' for r in records));del model;torch.cuda.empty_cache()
    metrics={n:summarize_rows(r,'region_r1',seed=20261011) for n,r in results.items()};baseline='native_real' if a.controls else 'native';paired={n:paired_difference(results[baseline],r,'region_r1',seed=20261011) for n,r in results.items()}
    write(out/'metrics.json',dict(models=metrics,paired_vs_native=paired,scope='external internal Dev used for epoch selection; region/tile multi-positive accuracy, not unique CityNav identity and not independent external Test'))

def main():
    p=argparse.ArgumentParser();p.add_argument('--features',required=True);p.add_argument('--heads',required=True);p.add_argument('--output',required=True);p.add_argument('--controls',action='store_true');p.add_argument('--black');run(p.parse_args())
if __name__=='__main__':main()
