"""Actual native ROI wall latency, including official preprocessing and transfers."""
import argparse
import gc
import importlib.metadata
import json
from pathlib import Path
import platform
import statistics
import time
import cv2
import numpy as np
from PIL import Image
import torch
from .city_features import CropEncoder,representations
from .compare_foundations import load_official
from .evaluate_features import load
from .goal_evidence import load_head
from .train_head import collate
from graphnav.renderer import CityNavObservedViewRenderer
from experiments.bidirectional_core.cache import DEFAULT_DATA


def measure(call,repeats=20):
    for _ in range(3):call()
    times=[]
    for _ in range(repeats):
        torch.cuda.synchronize();start=time.perf_counter();call();torch.cuda.synchronize();times.append((time.perf_counter()-start)*1000)
    return dict(repeats=repeats,mean_ms=statistics.mean(times),median_ms=statistics.median(times),p95_ms=float(np.quantile(times,.95)))


def main():
    p=argparse.ArgumentParser();p.add_argument('--features',required=True);p.add_argument('--assets',required=True);p.add_argument('--heads',required=True);p.add_argument('--output',required=True);p.add_argument('--data-root',default=DEFAULT_DATA);args=p.parse_args();output=Path(args.output)
    if output.exists():raise ValueError('refusing runtime overwrite')
    output.mkdir(parents=True);cv2.setNumThreads(1);start=time.time();rows=load(args.features);load_seconds=time.time()-start;row=rows[0];m=row['metadata'];public=row['public'];raw=json.loads((Path(args.data_root)/'cityrefer/objects.json').read_text());renderer=CityNavObservedViewRenderer(Path(args.data_root)/'rgbd',output/'renderer_private',image_size=384);start=time.time();obs=renderer.current_rgbd(m['map_name'],np.asarray(m['actual_camera_pose'],np.float64),float(public['ground_level']));renderer_seconds=time.time()-start
    crops=[representations(obs['bgr'],r,dict(id=r['object_id'],contour=raw[m['map_name']][r['object_id']]['contour']),obs['footprint_world'])[0] for r in m['roi_objects']];n=len(crops);result={}
    encoder=CropEncoder();torch.cuda.reset_peak_memory_stats()
    result['siglip2']=dict(parameters=sum(p.numel() for p in encoder.native.model.parameters()),single_roi=measure(lambda:encoder.encode(crops[:1])),observed_gallery=measure(lambda:encoder.encode(crops)),gallery_rois=n,peak_gpu_bytes=torch.cuda.max_memory_allocated())
    del encoder;gc.collect();torch.cuda.empty_cache()
    for name,relative in [('georsclip','georsclip/ckpt/RS5M_ViT-B-32.pt'),('skyclip','skyclip/SkyCLIP_ViT_B32_top50pct/epoch_20.pt')]:
        model,preprocess,_,quick=load_official(Path(args.assets)/relative,name);torch.cuda.reset_peak_memory_stats()
        def call(views):
            batch=torch.stack([preprocess(Image.fromarray(cv2.cvtColor(v,cv2.COLOR_BGR2RGB))) for v in views]).cuda()
            with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):return model.encode_image(batch)
        result[name]=dict(parameters=sum(p.numel() for p in model.parameters()),single_roi=measure(lambda:call(crops[:1])),observed_gallery=measure(lambda:call(crops)),gallery_rois=n,activation='QuickGELU' if quick else 'GELU',peak_gpu_bytes=torch.cuda.max_memory_allocated())
        del model;gc.collect();torch.cuda.empty_cache()
    for name,path in json.loads(Path(args.heads).read_text()).items():
        if path=='native':continue
        model=load_head(path,'cuda');p,_,_=collate([row],'polygon_masked','phrase');p={k:v.cuda() for k,v in p.items()};torch.cuda.reset_peak_memory_stats()
        with torch.inference_mode():latency=measure(lambda:model(p))
        result[name]=dict(parameters=sum(p.numel() for p in model.parameters()),cached_gallery_head_only=latency,gallery_rois=n,peak_gpu_bytes=torch.cuda.max_memory_allocated())
        del model,p;gc.collect();torch.cuda.empty_cache()
    versions={name:importlib.metadata.version(name) for name in ['torch','transformers','open_clip_torch','numpy','huggingface-hub']};versions['opencv']=cv2.__version__
    manifest=dict(models=result,real_roi_count=n,source_sample_id=m['sample_id'],source_current_cutoff=m['cutoff_step'],gpu=torch.cuda.get_device_name(),python=platform.python_version(),versions=versions,cuda=torch.version.cuda,feature_load_seconds=load_seconds,renderer_first_view_seconds=renderer_seconds,scope='single actual current 384-source view; warmed wall latency includes image preprocess, transfer, encoding; head timing on already-GPU frozen features excludes vision encoder; no closed-loop timing claim')
    (output/'metrics.json').write_text(json.dumps(manifest,indent=2));print(json.dumps(manifest,indent=2))


if __name__=='__main__':main()
