"""TRAIN-only online incremental latency/memory microbenchmark, no training."""
import argparse,json,time
from pathlib import Path
import cv2,numpy as np,torch
from graphnav.checkpointing import atomic_json
from graphnav.native_object_runtime import NativeObjectEncoder
from graphnav.landmark_projection import frame_to_world
from graphnav.map_guided_grounding import (public_landmarks,project_polygon,reference_candidates,
    crop_reference,reference_crop_bounds,reference_caption)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['train-records','catalog','output']:p.add_argument('--'+name,required=True)
    a=p.parse_args()
    if Path(a.output).exists():raise ValueError('Do not overwrite timing')
    records=[json.loads(line) for line in Path(a.train_records).read_text().splitlines()]
    catalog=json.loads(Path(a.catalog).read_text());selected={}
    for r in records:
        count=len(r['reference_ids'])
        if count not in selected:selected[count]=r
    for record in records:
        manifest=Path(record['raw_frame']).parent/'manifest.json'
        if json.loads(manifest.read_text()).get('partition')!='train':
            raise ValueError('Profiling is restricted to original TRAIN observations')
    model=NativeObjectEncoder('cuda')
    weights=sum(v.numel()*v.element_size() for v in model.model.parameters())+sum(v.numel()*v.element_size() for v in model.model.buffers())
    output=[]
    for count,r in sorted(selected.items()):
        frame=torch.load(r['raw_frame'],weights_only=True);bgr=cv2.imread(str(Path(r['raw_frame']).with_suffix('.png')))
        landmarks=public_landmarks(catalog,r['map'])
        observation=dict(bgr=bgr,depth_valid=frame['depth_valid'].numpy(),footprint_world=frame['footprint_world'].numpy())
        model.encode(observation,r['instruction']);torch.cuda.synchronize()
        durations=[];base_times=[];cpu_times=[];peaks=[]
        for _ in range(3):
            torch.cuda.synchronize();start=time.perf_counter();model.encode(observation,r['instruction']);torch.cuda.synchronize();base_times.append(time.perf_counter()-start)
            cpu_start=time.perf_counter()
            projections=[project_polygon(e['geometry_m'],observation['footprint_world'],bgr.shape,observation['depth_valid']) for e in landmarks]
            _,_,_,refs=reference_candidates(landmarks,projections,r['instruction'])
            cpu_times.append(time.perf_counter()-cpu_start)
            torch.cuda.synchronize();baseline=torch.cuda.memory_allocated();torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
            for index in refs:
                x0,y0,x1,y1=reference_crop_bounds(bgr.shape,projections[index]);crop=crop_reference(bgr,projections[index])
                corners=frame_to_world(np.array([[x0,y0],[x1-1,y0],[x1-1,y1-1],[x0,y1-1]],np.float32),observation['footprint_world'],bgr.shape)
                obs=dict(bgr=crop,depth_valid=observation['depth_valid'][y0:y1,x0:x1],footprint_world=corners)
                model.encode(obs,reference_caption(landmarks[index]))
            torch.cuda.synchronize();durations.append(time.perf_counter()-start);peaks.append((torch.cuda.max_memory_allocated()-baseline)/1024**2)
        output.append(dict(episode=r['episode'],step=r['step'],references=count,baseline_current_encoder_seconds=float(np.mean(base_times)),
                           projection_cpu_seconds=float(np.mean(cpu_times)),additional_reference_seconds=float(np.mean(durations)),
                           additional_peak_MiB=max(peaks),repetitions=3))
    atomic_json(dict(device=torch.cuda.get_device_name(0),torch=torch.__version__,frozen_model_resident_MiB=weights/1024**2,
                     cases=output,scope='TRAIN only; same encoder reused; warm3 repetitions; no PNG IO/model loading in timings; projectionCPU+referenceROI added; scoring/kernel overhead not included'),Path(a.output))


if __name__=='__main__':main()
