"""One native384 frozen vision forward per current frame; no private labels."""
import argparse
import json
from pathlib import Path
import time
import cv2
import numpy as np
import torch
from torch.nn import functional as F
from graphnav.appearance_matching import ObservedSigLIPMatchingEncoder
from graphnav.candidate_geometry import relative_entity_features
from graphnav.checkpointing import atomic_json,atomic_save,file_sha256
from graphnav.native_dense_regions import native_dense_regions,sparse_view,SPARSE_IDS
from graphnav.observed_patch_scales import masked_native_pool
from graphnav.visual_chain_position import WorldLikelihoodPositionHead


def geometry(world,entities):
    if not entities:return np.zeros((len(world),31),np.float32)
    pair=relative_entity_features(world,[dict(kind=x['kind'],geometry_m=x['geometry_m'].numpy()) for x in entities],len(entities))
    return np.concatenate((pair.mean(1),pair.min(1),pair.max(1),np.full((len(world),1),len(entities)/16)),axis=-1).astype(np.float32)


@torch.inference_mode()
def pool(native,patches,masks):
    flat=masks.reshape(-1,576);valid=flat.any(-1);result=torch.zeros(len(flat),1024,device='cuda')
    with torch.autocast('cuda',dtype=torch.bfloat16):
        if valid.any():result[torch.from_numpy(valid).cuda()]=F.normalize(masked_native_pool(native.model.vision_model.head,patches,torch.from_numpy(flat[valid]).cuda()).float(),dim=-1)
    return result.reshape(len(masks),3,1024).cpu().half().clone()


@torch.inference_mode()
def predict(model,q,region,features,control):
    image=features.clone()
    if control=='shuffle':
        ids=np.flatnonzero(region['valid']);image[ids]=image[ids[::-1].copy()].clone()
    bounds=q['world_bounds'].numpy();uv=(region['world']-bounds[:2])/(bounds[2:]-bounds[:2])
    x=dict(prior_logits=q['prior_logits'][None].cuda(),local_uv=torch.from_numpy(uv)[None].cuda(),image_features=image[None].cuda(),
        target_text=q['target_text'][None].cuda(),context_text=q['context_text'][None].cuda(),
        public_geometry=torch.from_numpy(geometry(region['world'],q['public_entities'])).half()[None].cuda(),
        quality=torch.from_numpy(region['quality'])[None].cuda(),local_valid=torch.from_numpy(region['valid'])[None].cuda())
    torch.cuda.synchronize();start=time.perf_counter();packed,visual=model(**x,return_visual=True);torch.cuda.synchronize()
    cosine=(F.normalize(image[:,0].float(),dim=-1)*F.normalize(q['target_text'].float(),dim=-1)).sum(-1)
    return dict(logprob=packed[0,:784].log_softmax(-1).cpu(),visual_score=visual[0].cpu().float(),cosine=cosine,
        world=torch.from_numpy(region['world'].copy()),pixels=torch.from_numpy(region['pixels'].copy()),valid=torch.from_numpy(region['valid'].copy()),
        head_seconds=time.perf_counter()-start)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('selection','checkpoint','backbone','output'):p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    m=json.loads((a.selection/'manifest.json').read_text());assert m['all_train_fit'] and file_sha256(a.selection/'inputs_public.pt')==m['public_inputs_sha256']
    queries=torch.load(a.selection/'inputs_public.pt',weights_only=True);torch.set_num_threads(4)
    a.output.mkdir(parents=True)
    native=ObservedSigLIPMatchingEncoder(str(a.backbone));head=WorldLikelihoodPositionHead().cuda().eval().requires_grad_(False)
    state=torch.load(a.checkpoint,weights_only=True,map_location='cpu');head.load_state_dict(state['model'])
    atomic_json(dict(checkpoint_sha256=file_sha256(a.checkpoint),public_selection_sha256=file_sha256(a.selection/'manifest.json'),
        backbone_weights_sha256={p.name:file_sha256(p) for p in a.backbone.glob('*.safetensors')},producer_sha256=file_sha256(__file__),
        region_code_sha256=file_sha256(Path('graphnav/native_dense_regions.py')),camera_yaw_rad=0.,private_labels_read=False,
        methods=['Native144','Native576','Native144Shuffle','Native576Shuffle']),a.output/'run_manifest.json')
    predictions={};runtime=[]
    with torch.inference_mode():
        for key,q in queries.items():
            frame=q['frame'];obs=Path(q['observations']);image=obs/frame['image'];depth=obs/frame['depth']
            assert file_sha256(image)==frame['image_sha256'] and file_sha256(depth)==frame['depth_sha256']
            bgr=cv2.imread(str(image))
            with np.load(depth) as d:valid=d['depth_valid'].copy()
            dense=native_dense_regions(dict(bgr=bgr,depth_valid=valid,footprint_world=frame['footprint_world']));sparse=sparse_view(dense)
            assert np.array_equal(sparse['world'],q['cached_world'].numpy()) and np.array_equal(sparse['valid'],q['cached_valid'].numpy())
            assert np.allclose(sparse['quality'],q['cached_quality'].numpy(),atol=1e-7)
            torch.cuda.reset_peak_memory_stats();t=time.perf_counter();pixels=native.processor(images=[cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)],return_tensors='pt').to('cuda');torch.cuda.synchronize();prep=time.perf_counter()-t
            assert list(pixels['pixel_values'].shape)==[1,3,384,384]
            t=time.perf_counter()
            with torch.autocast('cuda',dtype=torch.bfloat16):patches=native.model.vision_model(**pixels).last_hidden_state[0]
            torch.cuda.synchronize();encoder=time.perf_counter()-t
            t=time.perf_counter();features=pool(native,patches,dense['masks']);torch.cuda.synchronize();dense_time=time.perf_counter()-t
            t=time.perf_counter();timing_sparse=pool(native,patches,sparse['masks']);torch.cuda.synchronize();sparse_time=time.perf_counter()-t
            shared=features[torch.from_numpy(SPARSE_IDS)];similarity=F.cosine_similarity(shared.float(),timing_sparse.float(),dim=-1)
            predictions[key]={}
            for name,region,f,c in [('Native144',sparse,shared,'real'),('Native576',dense,features,'real'),('Native144Shuffle',sparse,shared,'shuffle'),('Native576Shuffle',dense,features,'shuffle')]:
                predictions[key][name]=predict(head,q,region,f,c)
            runtime.append(dict(frame_key=key,preprocess_seconds=prep,encoder_seconds=encoder,pool144_seconds=sparse_time,pool576_seconds=dense_time,
                peak_gpu_gib=torch.cuda.max_memory_allocated()/2**30,pooling_shape_cosine_min=float(similarity.min()),
                input_shape=[1,3,384,384],image_sha256=frame['image_sha256'],shared_prediction_features_exact=True,
                head_seconds={name:v['head_seconds'] for name,v in predictions[key].items()}))
            print(json.dumps(dict(completed=len(runtime),total=len(queries),frame_key=key)),flush=True)
    atomic_save(predictions,a.output/'predictions_public.pt');atomic_json(runtime,a.output/'runtime.json')
    atomic_json(dict(complete=True,cases=len(queries),predictions_sha256=file_sha256(a.output/'predictions_public.pt'),
        runtime_sha256=file_sha256(a.output/'runtime.json'),private_labels_read=False),a.output/'prediction_seal.json')

if __name__=='__main__':main()
