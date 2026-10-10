"""Fixed real-ROI BF16/FP32 checks; this script does not tune benchmark settings."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import time
import cv2
import numpy as np
import torch
from PIL import Image

MIN_FEATURE_COSINE = .995
MAX_SCORE_ABS_ERROR = .01
NORM_ATOL = .002


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def compare_features(image_fp32,image_bf16,text_fp32,text_bf16):
    """Evaluate paired normalized embeddings with tolerances declared above."""
    tensors=[torch.as_tensor(v,dtype=torch.float32).cpu() for v in
             (image_fp32,image_bf16,text_fp32,text_bf16)]
    ip,ib,tp,tb=tensors
    finite=all(bool(torch.isfinite(x).all()) for x in tensors)
    normalized=all(bool(torch.allclose(x.norm(dim=-1),torch.ones(len(x)),atol=NORM_ATOL)) for x in tensors)
    image_cos=(ip*ib).sum(-1);text_cos=(tp*tb).sum(-1)
    score_fp=ip@tp.T;score_bf=ib@tb.T;delta=(score_fp-score_bf).abs()
    top_fp=score_fp.argmax(-1);top_bf=score_bf.argmax(-1)
    top_values=score_fp.topk(min(2,score_fp.shape[1]),dim=-1).values
    gaps=(top_values[:,0]-top_values[:,1]) if score_fp.shape[1]>1 else torch.zeros(len(ip))
    result=dict(finite=finite,normalized=normalized,
        image_cosine_min=float(image_cos.min()),image_cosine_mean=float(image_cos.mean()),
        text_cosine_min=float(text_cos.min()),text_cosine_mean=float(text_cos.mean()),
        score_abs_error_max=float(delta.max()),score_abs_error_mean=float(delta.mean()),
        image_pair_cosines=image_cos.tolist(),text_pair_cosines=text_cos.tolist(),
        top1_text_rank_changes=int((top_fp!=top_bf).sum()),
        fp32_top1_text_indices=top_fp.tolist(),bf16_top1_text_indices=top_bf.tolist(),
        fp32_top1_top2_score_gaps=gaps.tolist(),
        score_fp32=score_fp.tolist(),score_bf16=score_bf.tolist())
    result['passed']=finite and normalized and min(result['image_cosine_min'],result['text_cosine_min'])>=MIN_FEATURE_COSINE and result['score_abs_error_max']<=MAX_SCORE_ABS_ERROR
    return result


@torch.inference_mode()
def direct(encoder,images,texts,model_name,dtype,microbatch):
    model=encoder.model;device=next(model.parameters()).device
    pieces=[];raw_image_dtypes=[];raw_text_dtypes=[]
    for start in range(0,len(images),microbatch):
        batch=images[start:start+microbatch]
        if model_name.startswith('remoteclip'):
            x=torch.stack([encoder.preprocess(Image.fromarray(v[:,:,::-1].copy())) for v in batch]).to(device)
        else:
            x=encoder.native.processor(images=[cv2.cvtColor(v,cv2.COLOR_BGR2RGB) for v in batch],return_tensors='pt').to(device)
        with torch.autocast('cuda',dtype=torch.bfloat16,enabled=dtype=='bf16'):
            feat=model.encode_image(x) if model_name.startswith('remoteclip') else model.get_image_features(**x)
        if not torch.is_tensor(feat):feat=feat.pooler_output
        raw_image_dtypes.append(str(feat.dtype));pieces.append(torch.nn.functional.normalize(feat.float(),dim=-1).cpu())
    if model_name.startswith('remoteclip'):
        tokens=encoder.tokenizer(texts).to(device)
    else:
        tokens=encoder.native.processor(text=texts,padding='max_length',truncation=True,
            max_length=model.config.text_config.max_position_embeddings,return_tensors='pt').to(device)
    with torch.autocast('cuda',dtype=torch.bfloat16,enabled=dtype=='bf16'):
        feat=model.encode_text(tokens) if model_name.startswith('remoteclip') else model.get_text_features(**tokens)
    if not torch.is_tensor(feat):feat=feat.pooler_output
    raw_text_dtypes.append(str(feat.dtype));language=torch.nn.functional.normalize(feat.float(),dim=-1).cpu()
    torch.cuda.synchronize()
    return torch.cat(pieces),language,dict(image_raw_dtypes=raw_image_dtypes,text_raw_dtypes=raw_text_dtypes,
        normalized_dtype='torch.float32',autocast=dtype=='bf16',weight_dtype=str(next(model.parameters()).dtype))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',required=True,choices=['remoteclip_vitl14','remoteclip_vitb32','siglip2_large384'])
    p.add_argument('--frozen-index',required=True);p.add_argument('--checkpoint');p.add_argument('--output',required=True)
    p.add_argument('--count',type=int,default=16);p.add_argument('--microbatch',type=int,default=4)
    a=p.parse_args();frozen=Path(a.frozen_index);timer=time.perf_counter()
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    protocol=json.loads((frozen/'frozen.json').read_text())
    for key in ('index','queries'):
        if sha(frozen/(key+'.jsonl'))!=protocol['files'][key]:raise ValueError('Frozen input changed')
    index=[json.loads(s) for s in (frozen/'index.jsonl').read_text().splitlines()][:a.count]
    queries=[json.loads(s) for s in (frozen/'queries.jsonl').read_text().splitlines()][:8]
    texts=[q[key] for q in queries for key in ('instruction','attribute_phrase')]
    images=[];input_provenance=[]
    for row in index:
        if row['observed_step']>row['observation_cutoff_step']:raise ValueError('Future ROI')
        image=cv2.imread(row['roi_path'])
        if image is None:raise FileNotFoundError(row['roi_path'])
        if sha(row['roi_path'])!=row['roi_sha256']:raise ValueError('ROI bytes changed')
        pixel=hashlib.sha256(str(image.shape).encode()+np.ascontiguousarray(image).tobytes()).hexdigest()
        images.append(image);input_provenance.append(dict(sample_id=row['sample_id'],roi_path=row['roi_path'],
            roi_file_sha256=row['roi_sha256'],pixel_bgr_sha256=pixel,pixel_shape=list(image.shape),
            observed_step=row['observed_step'],observation_cutoff_step=row['observation_cutoff_step']))
    if len(index)!=a.count:raise ValueError('Insufficient frozen real ROI count')
    if a.model.startswith('remoteclip'):
        from experiments.visual_grounding.adapters.remoteclip import Encoder
        encoder=Encoder('ViT-L-14' if a.model.endswith('l14') else 'ViT-B-32',checkpoint=a.checkpoint,batch_size=a.microbatch)
    else:
        from experiments.visual_grounding.adapters.siglip2 import Encoder
        encoder=Encoder(microbatch=a.microbatch)
    torch.cuda.reset_peak_memory_stats()
    ib,tb,bf_info=direct(encoder,images,texts,a.model,'bf16',a.microbatch)
    ip,tp,fp_info=direct(encoder,images,texts,a.model,'fp32',a.microbatch)
    direct_check=compare_features(ip,ib,tp,tb)
    # The direct BF16 check also verifies the actual production adapter path.
    adapter_i=torch.as_tensor(encoder.images(images),dtype=torch.float32).cpu()
    adapter_t=torch.as_tensor(encoder.texts(texts),dtype=torch.float32).cpu()
    result=compare_features(ip,adapter_i,tp,adapter_t)
    result['direct_bf16_fp32_check']=direct_check
    exact=torch.equal(adapter_i,ib) and torch.equal(adapter_t,tb)
    adapter_check=compare_features(ib,adapter_i,tb,adapter_t)
    result['production_adapter_bf16_exact_equivalence']=exact
    result['production_adapter_bf16_equivalence_passed']=adapter_check['passed']
    result['production_adapter_image_cosine_min']=adapter_check['image_cosine_min']
    result['production_adapter_text_cosine_min']=adapter_check['text_cosine_min']
    result['passed']=result['passed'] and adapter_check['passed']
    result.update(model=a.model,frozen_sha256=sha(frozen/'frozen.json'),inputs=input_provenance,texts=texts,
        selection='first 16 frozen real ROIs and first 8 language queries; fixed before precision results',
        thresholds=dict(feature_cosine_min=MIN_FEATURE_COSINE,score_abs_error_max=MAX_SCORE_ABS_ERROR,
                        normalized_atol=NORM_ATOL),bf16=bf_info,fp32=fp_info,
        torch=torch.__version__,gpu=torch.cuda.get_device_name(),microbatch=a.microbatch,
        peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,seconds=time.perf_counter()-timer,
        model_revision=getattr(getattr(encoder.model,'config',None),'_commit_hash',None),
        checkpoint_sha256=sha(a.checkpoint) if a.checkpoint else None,
        note='Rank changes near ties are reported, without changing benchmark precision or thresholds.')
    out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('model','passed','image_cosine_min','text_cosine_min',
        'score_abs_error_max','top1_text_rank_changes','production_adapter_bf16_exact_equivalence','seconds','peak_allocated_gib')},indent=2))
    if not result['passed']:raise SystemExit('Fixed precision tolerances failed')


if __name__=='__main__':main()
