"""Real ROI GeoRSCLIP/SkyCLIP native image-text comparison, same identities."""
import argparse
import json
from pathlib import Path
import time
import cv2
import numpy as np
from PIL import Image
import torch
import open_clip
from .city_features import representations
from .evaluate_features import load,metrics
from graphnav.renderer import CityNavObservedViewRenderer
from experiments.bidirectional_core.cache import DEFAULT_DATA
from experiments.bidirectional_core.stats import summarize_rows
from graphnav.checkpointing import file_sha256


def load_official(checkpoint,name):
    # Official GeoRSCLIP B/32 starts from OpenAI CLIP (QuickGELU); loading
    # identical parameter shapes into ordinary GELU would not be equivalent.
    # SkyCLIP B/32 starts from LAION2B e16 and uses GELU. Its official
    # notebook enables QuickGELU only for OpenAI-initialized L/14.
    quick_gelu=name=='georsclip'
    model,_,preprocess=open_clip.create_model_and_transforms('ViT-B-32',pretrained=None,force_quick_gelu=quick_gelu)
    # Official SkyCLIP optimizer metadata contains NumPy scalar values. Retain
    # the restricted unpickler; permit numeric types rather than arbitrary code.
    numeric_types=[np.core.multiarray.scalar,np.dtype,type(np.dtype('float64')),type(np.dtype('float32'))]
    with torch.serialization.safe_globals(numeric_types):
        state=torch.load(checkpoint,map_location='cpu',weights_only=True)
    if 'state_dict' in state:state=state['state_dict']
    state={k.removeprefix('module.'):v for k,v in state.items()}
    model.load_state_dict(state,strict=True);model.to('cuda').eval().requires_grad_(False)
    tokenizer=open_clip.get_tokenizer('ViT-B-32')
    return model,preprocess,tokenizer,quick_gelu


def run(args):
    output=Path(args.output)
    if output.exists():raise ValueError('refusing foundation comparison overwrite')
    output.mkdir(parents=True);cv2.setNumThreads(1);started=time.time();rows=load(args.features)
    model,preprocess,tokenizer,quick_gelu=load_official(args.checkpoint,args.model)
    raw=json.loads((Path(args.data_root)/'cityrefer/objects.json').read_text());renderer=CityNavObservedViewRenderer(Path(args.data_root)/'rgbd',output/'renderer_private',image_size=384)
    records={name+'_'+language:[] for name in ['tight','polygon_masked','context','multiscale','whole'] for language in ['full','phrase']}
    cached=[]
    for i,row in enumerate(rows):
        m=row['metadata'];o=renderer.current_rgbd(m['map_name'],np.asarray(m['actual_camera_pose'],np.float64),float(row['public']['ground_level']))
        crops=[representations(o['bgr'],r,dict(id=r['object_id'],contour=raw[m['map_name']][r['object_id']]['contour']),o['footprint_world']) for r in m['roi_objects']]
        variants={name:[c[j] for c in crops] for j,name in enumerate(['tight','polygon_masked','context'])};variants['whole']=[o['bgr']]
        embeddings={}
        with torch.inference_mode():
            for name,views in variants.items():
                batch=torch.stack([preprocess(Image.fromarray(cv2.cvtColor(v,cv2.COLOR_BGR2RGB))) for v in views]).cuda()
                with torch.autocast('cuda',dtype=torch.float16):result=model.encode_image(batch)
                embeddings[name]=torch.nn.functional.normalize(result.float(),dim=-1).cpu().half()
            embeddings['whole']=embeddings['whole'].expand(len(crops),-1).clone()
            embeddings['multiscale']=torch.nn.functional.normalize(embeddings['tight'].float()+embeddings['context'].float(),dim=-1).half()
            texts=tokenizer([m['instruction'],row['phrase']]).cuda()
            with torch.autocast('cuda',dtype=torch.float16):text=model.encode_text(texts)
            text=torch.nn.functional.normalize(text.float(),dim=-1).cpu().half()
        if not all(torch.isfinite(v).all() for v in [*embeddings.values(),text]):
            raise ValueError('nonfinite official CUDA embeddings')
        if i==0:
            with torch.inference_mode(),torch.autocast('cuda',enabled=False):
                reference_images=torch.stack([preprocess(Image.fromarray(cv2.cvtColor(v,cv2.COLOR_BGR2RGB))) for v in variants['tight']]).cuda()
                reference_images=torch.nn.functional.normalize(model.encode_image(reference_images).float(),dim=-1).cpu()
                reference_text=torch.nn.functional.normalize(model.encode_text(texts).float(),dim=-1).cpu()
            image_cos=torch.nn.functional.cosine_similarity(embeddings['tight'].float(),reference_images,dim=-1)
            text_cos=torch.nn.functional.cosine_similarity(text.float(),reference_text,dim=-1)
            error=float((embeddings['tight'].float()@text.float().T-reference_images@reference_text.T).abs().max())
            precision=dict(actual_rois=len(crops),minimum_image_cosine=float(image_cos.min()),minimum_text_cosine=float(text_cos.min()),maximum_score_error=error,
                thresholds=dict(cosine=.995,score_error=.01),passed=bool(image_cos.min()>=.995 and text_cos.min()>=.995 and error<=.01))
            (output/'precision_gate.json').write_text(json.dumps(precision,indent=2))
            if not precision['passed']:raise ValueError('official model real-ROI FP32 precision gate failed')
        for name,images in embeddings.items():
            for j,language in enumerate(['full','phrase']):
                score=images.float()@text[j].float();r=dict(sample_id=m['sample_id'],family_id=m['family_id'],map_name=m['map_name'],direction=m['direction'],metrics=metrics(score,row),scores=score.tolist())
                records[name+'_'+language].append(r)
        cached.append(dict(sample_id=m['sample_id'],features=embeddings,texts=text))
        if i%25==0:print(json.dumps(dict(states=i+1,total=len(rows),seconds=time.time()-started)),flush=True)
    torch.save(cached,output/'native_features.pt')
    summary={name:{key:summarize_rows(r,key,repetitions=2000) for key in r[0]['metrics']} for name,r in records.items()}
    for name,r in records.items():
        with (output/(name+'_predictions.jsonl')).open('w') as h:
            for item in r:h.write(json.dumps(item)+'\n')
    (output/'metrics.json').write_text(json.dumps(dict(model=args.model,models=summary,seconds=time.time()-started,peak_gpu_bytes=torch.cuda.max_memory_allocated(),checkpoint_sha256=file_sha256(args.checkpoint),strict_state_loading=True,image_text_same_checkpoint=True,activation='QuickGELU' if quick_gelu else 'GELU',feature_manifest_sha256=file_sha256(Path(args.features)/'manifest.json'),dimension=int(text.shape[-1]),preprocess=repr(preprocess)),indent=2))


def main():
    p=argparse.ArgumentParser();p.add_argument('--features',required=True);p.add_argument('--checkpoint',required=True);p.add_argument('--model',choices=['georsclip','skyclip'],required=True);p.add_argument('--output',required=True);p.add_argument('--data-root',default=DEFAULT_DATA);run(p.parse_args())


if __name__=='__main__':main()
