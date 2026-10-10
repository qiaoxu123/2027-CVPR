"""Actual RGB interventions preserving each destination's polygon and geometry."""
import argparse
import json
from pathlib import Path
import time
import cv2
import numpy as np
import torch
from torch.nn import functional as F
from .city_features import CropEncoder,representations
from .evaluate_features import load
from experiments.bidirectional_core.cache import DEFAULT_DATA
from graphnav.renderer import CityNavObservedViewRenderer
from graphnav.landmark_projection import project_landmark
from graphnav.checkpointing import file_sha256


def intervention(tight,mask,donor=None):
    """RGB changes inside the original observed polygon only."""
    pixels=np.zeros_like(tight) if donor is None else cv2.resize(donor,(tight.shape[1],tight.shape[0]))
    pixels[~mask]=127
    return pixels


def run(args):
    cv2.setNumThreads(1);output=Path(args.output)
    if output.exists():raise ValueError('refusing intervention overwrite')
    output.mkdir(parents=True);started=time.time();rows=load(args.features)
    encoder=CropEncoder();renderer=CityNavObservedViewRenderer(Path(args.data_root)/'rgbd',output/'renderer_private',image_size=384)
    raw=json.loads((Path(args.data_root)/'cityrefer/objects.json').read_text())
    files=[];chunk=[];changed=0;total=0;precision=None
    for index,row in enumerate(rows):
        m=row['metadata'];p=row['public'];regions=m['roi_objects']
        observation=renderer.current_rgbd(m['map_name'],np.asarray(m['actual_camera_pose'],np.float64),float(p['ground_level']))
        crops=[];masks=[]
        for region in regions:
            polygon=dict(id=region['object_id'],object_type=region['object_type'],contour=raw[m['map_name']][region['object_id']]['contour'])
            crops.append(representations(observation['bgr'],region,polygon,observation['footprint_world'])[0])
            projection=project_landmark(polygon,observation['footprint_world'],observation['bgr'].shape,min_visible=.9,min_pixels=256)
            a,b,c,d=map(int,region['bbox']);masks.append(projection.mask[b:d,a:c].astype(bool))
        donors=[]
        for j,region in enumerate(regions):
            peers=sorted((k for k,r in enumerate(regions) if r['object_type']==region['object_type']),key=lambda k:regions[k]['object_id'])
            donor=peers[(peers.index(j)+1)%len(peers)];donors.append(donor)
        zero=[intervention(c,m) for c,m in zip(crops,masks)]
        shuffled=[intervention(c,m,crops[k]) for c,m,k in zip(crops,masks,donors)]
        zv,zt=encoder.encode(zero,tokens=True);sv,st=encoder.encode(shuffled,tokens=True)
        if precision is None:
            # Real same-ROI FP32 comparison, not a generated smoke image.
            images=[intervention(c,m,c) for c,m in zip(crops,masks)][:16]
            actual,_=encoder.encode(images)
            batch=encoder.native.processor(images=[cv2.cvtColor(v,cv2.COLOR_BGR2RGB) for v in images],return_tensors='pt').to(encoder.device)
            with torch.inference_mode(),torch.autocast('cuda',enabled=False):
                hidden=encoder.native.model.vision_model(**batch).last_hidden_state
                reference=F.normalize(encoder.native.model.vision_model.head(hidden).float(),dim=-1).cpu()
            text=encoder.native.texts([m['instruction'],row['phrase']]).float().cpu()
            text_batch=encoder.native.processor(text=[m['instruction'],row['phrase']],padding='max_length',truncation=True,max_length=encoder.native.model.config.text_config.max_position_embeddings,return_tensors='pt').to(encoder.device)
            with torch.inference_mode(),torch.autocast('cuda',enabled=False):
                rt=F.normalize(encoder.native.model.text_model(**text_batch).pooler_output.float(),dim=-1).cpu()
            image_cos=F.cosine_similarity(actual.float(),reference).min().item();text_cos=F.cosine_similarity(text,rt).min().item()
            error=((actual.float()@text.T)-(reference@rt.T)).abs().max().item()
            precision=dict(actual_rois=len(images),minimum_image_cosine=image_cos,minimum_text_cosine=text_cos,maximum_score_error=error,passed=image_cos>=.995 and text_cos>=.995 and error<=.01)
            (output/'precision_gate.json').write_text(json.dumps(precision,indent=2))
            if not precision['passed']:raise ValueError('actual SigLIP2 crop precision gate failed')
        chunk.append(dict(sample_id=m['sample_id'],zero_rgb=zv,shuffled_rgb=sv,zero_tokens=zt,shuffled_tokens=st,donor_indices=donors,intervened=[j!=k for j,k in enumerate(donors)]))
        total+=len(donors);changed+=sum(j!=k for j,k in enumerate(donors))
        if len(chunk)==16 or index==len(rows)-1:
            path=output/f'chunk_{len(files):04}.pt';torch.save(chunk,path);files.append(path.name);chunk=[]
        if index%50==0:print(json.dumps(dict(states=index+1,total=len(rows),seconds=time.time()-started)),flush=True)
    manifest=dict(complete=True,chunks=files,files={p:file_sha256(output/p) for p in files},features_manifest_sha256=file_sha256(Path(args.features)/'manifest.json'),states=len(rows),rois=total,intervened_rois=changed,seconds=time.time()-started,peak_gpu_bytes=torch.cuda.max_memory_allocated(),policy='same-class cyclic donor from current real observed view, public object order; destination polygon, pose, quality and language unchanged; singleton objects unchanged and flagged; zero RGB inside and 127 outside original mask',precision=precision)
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2));print(json.dumps(manifest),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--features',required=True);p.add_argument('--output',required=True);p.add_argument('--data-root',default=DEFAULT_DATA);run(p.parse_args())


if __name__=='__main__':main()
