"""Frozen GPU current-view tiling pilot. No GT label input is accepted."""
import argparse
import json
from pathlib import Path
import shutil
import sys
import time
import cv2
import numpy as np
import torch
from torch.nn import functional as F
from graphnav.appearance_matching import ObservedSigLIPMatchingEncoder
from graphnav.candidate_geometry import relative_entity_features
from graphnav.checkpointing import atomic_json,atomic_save,file_sha256
from graphnav.fixed_yaw_observation import FixedYawObservation
from graphnav.local_chain_observation import local_patch_regions
from graphnav.observed_patch_scales import masked_native_pool
from graphnav.observed_tile_regions import split_current_tiles,tile_region_inputs
from graphnav.renderer import CityNavObservedViewRenderer
from graphnav.visual_chain_position import WorldLikelihoodPositionHead


@torch.inference_mode()
def encode(native,images,masks,*,only_first=False):
    torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();started=time.perf_counter()
    pixels=native.processor(images=[cv2.cvtColor(x,cv2.COLOR_BGR2RGB) for x in images],return_tensors='pt').to(native.device)
    assert tuple(pixels['pixel_values'].shape)==(len(images),3,384,384)
    shape=tuple(masks.shape[:2]);flat=masks.reshape(-1,masks.shape[-1]);supported=flat.any(-1)
    with torch.autocast('cuda',dtype=torch.bfloat16):
        encoded=native.model.vision_model(**pixels).last_hidden_state
        tokens=encoded[0] if only_first else encoded.reshape(-1,native.dimension)
        if tokens.shape[0]!=flat.shape[-1]:raise ValueError('tile-major tokens and physical masks differ')
        values=torch.zeros(len(flat),native.dimension,device=native.device,dtype=torch.float32)
        if supported.any():
            pooled=masked_native_pool(native.model.vision_model.head,tokens,torch.from_numpy(flat[supported]).to(native.device))
            values[torch.from_numpy(supported).to(native.device)]=F.normalize(pooled.float(),dim=-1)
    features=values.reshape(*shape,native.dimension).cpu().half().clone()
    torch.cuda.synchronize()
    return features,dict(seconds=time.perf_counter()-started,gpu_peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
        network_input_shape=list(pixels['pixel_values'].shape),tokens=int(tokens.shape[0]))


def geometry(world,entities):
    if not entities:return np.zeros((len(world),31),np.float32)
    pair=relative_entity_features(world,entities,len(entities))
    return np.concatenate((pair.mean(1),pair.min(1),pair.max(1),np.full((len(world),1),len(entities)/16,np.float32)),-1)


@torch.inference_mode()
def predict(head,query,region,features):
    bounds=np.asarray(query['world_bounds'],np.float32);world=region['world'];uv=(world-bounds[:2])/(bounds[2:]-bounds[:2])
    args=dict(prior_logits=query['prior_logits'][None].cuda(),local_uv=torch.from_numpy(uv)[None].cuda(),image_features=features[None].cuda(),
        target_text=query['target_text'][None].cuda(),context_text=query['context_text'][None].cuda(),
        public_geometry=torch.from_numpy(geometry(world,query['public_entities'])).half()[None].cuda(),
        quality=torch.from_numpy(region['quality'])[None].cuda(),local_valid=torch.from_numpy(region['valid'])[None].cuda())
    torch.cuda.synchronize();started=time.perf_counter();packed,visual=head(**args,return_visual=True);logprobs=packed[0,:784].float().log_softmax(-1).cpu();torch.cuda.synchronize()
    valid=region['valid']&(uv>=0).all(-1)&(uv<=1).all(-1)
    native_similarity=(F.normalize(features[:,0].float(),dim=-1)*F.normalize(query['target_text'].float(),dim=-1)).sum(-1)
    choice=int(logprobs.argmax())
    return dict(goal_world=query['global_world'][choice].tolist(),global_choice=choice,goal_logprob=logprobs,
        local_world=torch.from_numpy(world.copy()),valid=torch.from_numpy(valid.copy()),visual_logits=visual[0].cpu().float(),
        native_similarity=native_similarity,head_seconds=time.perf_counter()-started,valid_candidates=int(valid.sum()))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--batch-audit',action='store_true')
    for key in ('selection','observations','rgbd','sbf-source','checkpoint','output'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--backbone',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError('preserve pilot; no outcome-based reruns')
    torch.set_num_threads(4);torch.manual_seed(0)
    manifest=json.loads((a.selection/'manifest.json').read_text())
    for name in ('public_selection','public_inputs'):
        suffix='.json' if name=='public_selection' else '.pt'
        if file_sha256(a.selection/(name+suffix))!=manifest[name+'_sha256']:raise ValueError('selected public inputs changed')
    selection=json.loads((a.selection/'public_selection.json').read_text());queries=torch.load(a.selection/'public_inputs.pt',map_location='cpu',weights_only=True)
    if len(selection)!=manifest['cases'] or not 1<=len(selection)<=64 or not manifest['all_fit']:raise ValueError('bounded frozen fit selection required')
    sys.path.insert(0,str(a.sbf_source));from sbfnav.mapdata import GROUND_LEVEL
    a.output.mkdir(parents=True);(a.output/'images').mkdir();(a.output/'features').mkdir()
    native=ObservedSigLIPMatchingEncoder(str(a.backbone));head=WorldLikelihoodPositionHead().cuda().eval().requires_grad_(False)
    state=torch.load(a.checkpoint,map_location='cpu',weights_only=True)
    if state['architecture']!='world-likelihood-v3':raise ValueError('original frozen V3 required')
    head.load_state_dict(state['model'],strict=True)
    renderer=FixedYawObservation(CityNavObservedViewRenderer(a.rgbd,a.output/'unused',image_size=768),0.)
    frozen=dict(checkpoint_sha256=file_sha256(a.checkpoint),selection_sha256=file_sha256(a.selection/'public_selection.json'),
        public_inputs_sha256=file_sha256(a.selection/'public_inputs.pt'),source_sha256=file_sha256(__file__),
        geometry_source_sha256=file_sha256(Path('graphnav/observed_tile_regions.py')),
        backbone=str(a.backbone),backbone_weights_sha256={x.name:file_sha256(x) for x in sorted(a.backbone.glob('*.safetensors'))},
        inference_reads_private_labels=False,camera_yaw_rad=0.,weights_frozen=True,methods=['Whole384_144','Interpolated768_144','Native768_144','Interpolated768_576','Native768_576'])
    if a.batch_audit:frozen['methods'].append('Whole384Repeat4_144')
    atomic_json(frozen,a.output/'run_manifest.json');predictions={};stats=[]
    dense_sparse_ids=np.array([y*24+x for y in range(1,24,2) for x in range(1,24,2)])
    for index,case in enumerate(selection):
        key=case['case_key'];query=queries[key];frame=query['frame'];image=a.observations/frame['image'];depth=a.observations/frame['depth']
        if file_sha256(image)!=frame['image_sha256'] or file_sha256(depth)!=frame['depth_sha256']:raise ValueError('original actual observation changed')
        low=dict(bgr=cv2.imread(str(image)),depth_valid=np.load(depth)['depth_valid'],footprint_world=np.asarray(frame['footprint_world']),pose=np.asarray(frame['camera_pose']))
        started=time.perf_counter();high=renderer.current_rgbd(case['map_name'],np.asarray(frame['agent_pose']),GROUND_LEVEL[case['map_name']]);render_seconds=time.perf_counter()-started
        dense=tile_region_inputs(low,high,dense=True);sparse=tile_region_inputs(low,high)
        if not np.allclose(sparse['world'],query['cached_world'].numpy(),atol=.001):raise ValueError('candidate positions changed in matched-resolution comparison')
        if not np.allclose(sparse['quality'],query['cached_quality'].numpy(),atol=1e-6):raise ValueError('quality inputs differ from original')
        if not np.array_equal(sparse['masks'],dense['masks'][dense_sparse_ids]):raise ValueError('sparse must be an exact subset')
        old_world,old_masks,old_valid,_=local_patch_regions(low)
        whole,whole_time=encode(native,[low['bgr']],old_masks)
        interp=cv2.resize(low['bgr'],(768,768),interpolation=cv2.INTER_LINEAR)
        inter_features,inter_time=encode(native,split_current_tiles(interp),dense['masks'])
        native_features,native_time=encode(native,split_current_tiles(high['bgr']),dense['masks'])
        # All 144 comparisons share the same geometry and common validity.
        methods={'Whole384_144':(sparse,whole),'Interpolated768_144':(sparse,inter_features[dense_sparse_ids]),
            'Native768_144':(sparse,native_features[dense_sparse_ids]),'Interpolated768_576':(dense,inter_features),'Native768_576':(dense,native_features)}
        repeat=None;repeat_time=None
        if a.batch_audit:
            repeat,repeat_time=encode(native,[low['bgr']]*4,old_masks,only_first=True)
            methods['Whole384Repeat4_144']=(sparse,repeat)
        predictions[key]={name:predict(head,query,region,features) for name,(region,features) in methods.items()}
        atomic_save(dict(whole384=whole,interpolated768=inter_features,native768=native_features,
            dense={k:torch.from_numpy(v.copy()) for k,v in dense.items() if k!='masks'},sparse_indices=torch.from_numpy(dense_sparse_ids),whole384_repeat4=repeat),a.output/'features'/(key+'.pt'))
        for name,rgb in [('low384',low['bgr']),('native768',high['bgr']),('interpolated768',interp)]:
            out=a.output/'images'/(key+'_'+name+'.png')
            if not cv2.imwrite(str(out),rgb):raise IOError(out)
        valid=torch.from_numpy(sparse['valid']);similarity=F.cosine_similarity(whole[valid].float(),query['cached_image_features'][valid].float(),dim=-1)
        stats.append(dict(case_key=key,map_name=case['map_name'],render_seconds=render_seconds,whole384=whole_time,interpolated768=inter_time,native768=native_time,
            old_valid=int(old_valid.sum()),common_valid_144=int(sparse['valid'].sum()),common_valid_576=int(dense['valid'].sum()),
            repeat4_runtime=repeat_time,
            repeat4_cache_cosine_mean=float(F.cosine_similarity(repeat[valid].float(),query['cached_image_features'][valid].float(),dim=-1).mean()) if repeat is not None and valid.any() else None,
            repeat4_cache_cosine_min=float(F.cosine_similarity(repeat[valid].float(),query['cached_image_features'][valid].float(),dim=-1).min()) if repeat is not None and valid.any() else None,
            cache_cosine_mean=float(similarity.mean()) if similarity.numel() else None,cache_cosine_min=float(similarity.min()) if similarity.numel() else None,
            footprint_world=np.asarray(high['footprint_world']).tolist(),camera_pose=np.asarray(high['pose']).tolist(),
            images={name:file_sha256(a.output/'images'/(key+'_'+name+'.png')) for name in ('low384','native768','interpolated768')}))
        atomic_json(stats,a.output/'runtime.json');print(json.dumps(dict(completed=index+1,total=len(selection),case_key=key,whole_seconds=whole_time['seconds'],native_seconds=native_time['seconds'])),flush=True)
    atomic_save(predictions,a.output/'predictions_public.pt')
    atomic_json(dict(complete=True,cases=len(selection),predictions_sha256=file_sha256(a.output/'predictions_public.pt'),runtime_sha256=file_sha256(a.output/'runtime.json'),
        run_manifest_sha256=file_sha256(a.output/'run_manifest.json'),GT_read=False),a.output/'prediction_seal.json')


if __name__=='__main__':main()
