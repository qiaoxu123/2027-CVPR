"""Measured cached-head runtime; do not claim full navigation latency."""
import argparse,json,time
from pathlib import Path
import torch
from .dataset import city_rows,collate
from .evaluate import load_model,predict_scores
from .data import write


def run(a):
    out=Path(a.output)
    if out.exists():raise ValueError('no overwrite')
    out.mkdir(parents=True);torch.set_num_threads(2);start=time.perf_counter();rows=city_rows(a.features);load_seconds=time.perf_counter()-start
    representative=next(r for r in rows if len(r['public']['roi_valid'])>0);p,_,_=collate([representative]);p={k:v.cuda() for k,v in p.items()};results={}
    for name,path in json.loads(Path(a.config).read_text())['heads'].items():
        model=load_model(path);begin=time.perf_counter()
        if model is None:
            def forward():return (torch.nn.functional.normalize(p['images'],dim=-1)*torch.nn.functional.normalize(p['full'],dim=-1)[:,None]).sum(-1)
        elif hasattr(model,'KEYS'):
            def forward():return model(p)
        else:
            q=dict(text=p['full'],images=p['images'],patch_tokens=p['patch_tokens'],language_tokens=p['language_tokens'],language_valid=p['language_valid'],quality=p['quality'],valid=p['valid'])
            def forward():return model(q)
        with torch.inference_mode():
            for i in range(10):forward()
            torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();times=[]
            for i in range(100):
                t=time.perf_counter();forward();torch.cuda.synchronize();times.append(time.perf_counter()-t)
        results[name]=dict(candidate_count=int(p['valid'].sum()),median_ms=float(torch.tensor(times).median())*1000,p95_ms=float(torch.quantile(torch.tensor(times),.95))*1000,peak_allocated_bytes=torch.cuda.max_memory_allocated(),iterations=100,seconds=time.perf_counter()-begin,scope='preloaded real causal feature candidate set; excludes encoding/rendering/IO/controller')
        del model;torch.cuda.empty_cache()
    write(out/'metrics.json',dict(models=results,feature_loading_seconds=load_seconds,seed=20261011,whole_navigation_latency_measured=False))

def main():
    p=argparse.ArgumentParser();p.add_argument('--features',required=True);p.add_argument('--config',required=True);p.add_argument('--output',required=True);run(p.parse_args())
if __name__=='__main__':main()
