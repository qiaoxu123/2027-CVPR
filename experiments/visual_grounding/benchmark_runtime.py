"""Real CUDA timing, with image feature caches disabled for fair encoder timing."""
import argparse,json,time
from pathlib import Path
import cv2
import numpy as np
import torch
from experiments.visual_grounding.evaluate_zero_shot import load_encoder,tensor,validate
from experiments.visual_grounding.freeze_dataset import read,sha


def main():
    p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--checkpoint');p.add_argument('--official-repo');p.add_argument('--data',required=True);p.add_argument('--output',required=True);p.add_argument('--device',default='cuda');p.add_argument('--batch-size',type=int,default=16);p.add_argument('--repeats',type=int,default=10);a=p.parse_args()
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True);encoder=load_encoder(a)
    # Runtime uses Train/Seen actual ROIs only, without reading Unseen scores.
    rows=[r for r in read(Path(a.data)/'index.jsonl') if r['split']!='val_unseen'][:16]
    timings={}
    for n in [1,16]:
        chosen=(rows*16)[:n];paths=[r['roi_source'] if a.model=='regionclip' else r['roi_path'] for r in chosen];images=[cv2.imread(p) for p in paths]
        if any(x is None for x in images):raise FileNotFoundError('actual CityNav ROI')
        samples=[];torch.cuda.reset_peak_memory_stats()
        for trial in range(a.repeats+2):
            if hasattr(encoder,'native'):encoder.native.cache.clear();encoder.native.text_cache.clear()
            if hasattr(encoder,'_frame_key'):encoder._frame_key=None
            torch.cuda.synchronize();t=time.perf_counter()
            result=encoder.regions(images,[r['roi_bbox'] for r in chosen]) if a.model=='regionclip' else encoder.images(images)
            encoder.texts(['a building with a gray roof']);torch.cuda.synchronize();elapsed=time.perf_counter()-t
            validate(tensor(result))
            if trial>=2:samples.append(elapsed)
        timings[str(n)]=dict(median_ms=float(np.median(samples)*1000),p95_ms=float(np.quantile(samples,.95)*1000),throughput_roi_s=n/float(np.median(samples)),peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,repetitions=a.repeats,images=n,texts=1,cache='cleared feature cache; loaded model and decoded pixels; 2 warmups')
    (out/'runtime_benchmark.json').write_text(json.dumps(dict(model=a.model,frozen_sha256=sha(Path(a.data)/'frozen.json'),results=timings,device=torch.cuda.get_device_name()),indent=2));print(json.dumps(timings))
if __name__=='__main__':main()
