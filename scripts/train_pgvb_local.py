"""Same PGVB architecture; local Anchor positives-only loss and fixed update budget."""
import argparse,gc,json,math,random,time
from pathlib import Path
import cv2
import numpy as np
import torch
from graphnav.checkpointing import atomic_json,atomic_save,file_sha256
from graphnav.polygon_guided_visual import PolygonGuidedVisualBelief
from graphnav.pgvb_local_training import pgvb_batch,local_pgvb_loss as pgvb_loss,frame_batches


def raw_smoke(data,model,backbone):
    from graphnav.appearance_matching import ObservedSigLIPMatchingEncoder
    from graphnav.native_dense_regions import native_dense_regions,sparse_view
    from scripts.run_native_grid_pilot import pool
    indices=next(frame_batches(data,'fit',1));r=data['records'][indices[0]];f=data['frames'][r['frame_id']]
    rgb=cv2.imread(f['image_path']);path=Path(f['image_path']).parent/f['frame']['depth']
    # Older observation files may include rgb/ prefix relative to their root.
    if not path.exists():path=Path(f['image_path']).parent/Path(f['frame']['depth']).name
    with np.load(path) as npz:depth=npz['depth_valid'].copy()
    dense=sparse_view(native_dense_regions(dict(bgr=rgb,depth_valid=depth,footprint_world=f['frame']['footprint_world'])))
    native=ObservedSigLIPMatchingEncoder(str(backbone));pixels=native.processor(images=[cv2.cvtColor(rgb,cv2.COLOR_BGR2RGB)],return_tensors='pt').to('cuda')
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):patch=native.model.vision_model(**pixels).last_hidden_state[0]
    features=pool(native,patch,dense['masks']).clone()
    batch,labels,rows=pgvb_batch(data,indices);batch['image_features']=features[None].repeat(len(indices),1,1,1).cuda().requires_grad_(True)
    loss=pgvb_loss(model(**batch),labels,rows);loss['total'].backward()
    gradients={k:float(p.grad.abs().sum()) if p.grad is not None else None for k,p in model.named_parameters()}
    result=dict(image_sha256=file_sha256(f['image_path']),actual_rgb_shape=list(rgb.shape),network_input_shape=list(pixels['pixel_values'].shape),
        loss=float(loss['total'].detach()),parameter_gradients=gradients,image_input_gradient=float(batch['image_features'].grad.abs().sum()),
        backbone_has_grad=any(p.grad is not None for p in native.model.parameters()),fresh_cached_feature_mae=float((features.float()-f['image_features'].float()).abs().mean()))
    if not math.isfinite(result['loss']) or not result['image_input_gradient']>0 or any(x is None or not math.isfinite(x) for x in gradients.values()) or result['backbone_has_grad']:raise ValueError('GPU smoke gradient/backbone failure')
    model.zero_grad(set_to_none=True);del native,pixels,features,batch,loss;gc.collect();torch.cuda.empty_cache()
    return result


def main():
    p=argparse.ArgumentParser()
    for k in ('data','output'):p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--max-updates',type=int,default=5112);p.add_argument('--epochs',type=int,default=12);p.add_argument('--smoke-steps',type=int);p.add_argument('--resume',action='store_true');p.add_argument('--raw-backbone',type=Path)
    a=p.parse_args();torch.set_num_threads(4);torch.manual_seed(0);rng=random.Random(0);np.random.seed(0)
    proof=json.loads((a.data/'manifest.json').read_text())
    if file_sha256(a.data/'cache.pt')!=proof['cache_sha256']:raise ValueError('Dataset changed')
    data=torch.load(a.data/'cache.pt',weights_only=True,map_location='cpu')
    if a.output.exists() and not a.resume:raise FileExistsError(a.output)
    a.output.mkdir(parents=True,exist_ok=True);model=PolygonGuidedVisualBelief().cuda()
    optimizer=torch.optim.AdamW(model.parameters(),lr=.0005,weight_decay=.001)
    code={str(p):file_sha256(p) for p in (Path(__file__),Path('graphnav/polygon_guided_visual.py'),Path('graphnav/pgvb_local_training.py'),Path('graphnav/local_landmark_anchor.py'))}
    steps_per_epoch=sum(1 for _ in frame_batches(data,'fit',4))
    config=dict(seed=0,epochs=a.epochs,max_updates=a.max_updates,steps_per_epoch=steps_per_epoch,lr=.0005,weight_decay=.001,batch_frames=4,hidden=64,heads=4,
        loss_weights=dict(region=1,bce=.2,same_image_pair=.3),parameters=sum(p.numel() for p in model.parameters()),
        data_manifest_sha256=file_sha256(a.data/'manifest.json'),code_sha256=code,backbone_frozen=True,LoRA=False,
        model_selection='lowest epoch TRAIN mean loss; held-out city never read',no_geometry_belief=True)
    epoch0=0;steps=0;history=[];best=float('inf');raw=None
    if a.resume:
        state=torch.load(a.output/'last.pt',map_location='cpu',weights_only=True)
        if state['config']!=config:raise ValueError('Resume configuration changed')
        model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer']);epoch0=state['next_epoch'];steps=state['steps'];history=state['history'];best=state['best'];rng.setstate(state['rng']);torch.set_rng_state(state['torch_rng']);torch.cuda.set_rng_state_all(state['cuda_rng']);raw=state['raw_smoke']
    elif a.raw_backbone:raw=raw_smoke(data,model,a.raw_backbone)
    atomic_json(config,a.output/'config.json')
    if raw:atomic_json(raw,a.output/'raw_gpu_smoke.json')
    started=time.perf_counter();torch.cuda.reset_peak_memory_stats();gradient_proof=None
    for epoch in range(epoch0,a.epochs):
        values=[]
        for indices in frame_batches(data,'fit',4,rng):
            batch,labels,rows=pgvb_batch(data,indices);scores=model(**batch);loss=pgvb_loss(scores,labels,rows)
            if not torch.isfinite(loss['total']):raise ValueError('Nonfinite training loss')
            optimizer.zero_grad(set_to_none=True);loss['total'].backward()
            if gradient_proof is None:
                gradient_proof={k:float(p.grad.abs().sum()) if p.grad is not None else None for k,p in model.named_parameters()}
                if any(x is None or not math.isfinite(x) for x in gradient_proof.values()):raise ValueError('Invalid trainable gradient')
            torch.nn.utils.clip_grad_norm_(model.parameters(),5);optimizer.step();steps+=1
            values.append({k:float(v.detach()) if torch.is_tensor(v) else float(v) for k,v in loss.items()})
            if (a.smoke_steps and steps>=a.smoke_steps) or steps>=a.max_updates:break
        avg={k:float(np.mean([v[k] for v in values])) for k in values[0]}
        history.append(dict(epoch=epoch+1,steps=steps,seconds=time.perf_counter()-started,fractional_epochs=steps/steps_per_epoch,completed_epoch=(len(values)==steps_per_epoch),**avg))
        improved=avg['total']<best;best=min(best,avg['total'])
        state=dict(config=config,model=model.state_dict(),optimizer=optimizer.state_dict(),next_epoch=epoch+1,steps=steps,history=history,best=best,rng=rng.getstate(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),raw_smoke=raw)
        atomic_save(state,a.output/'last.pt')
        if improved:atomic_save(state,a.output/'best.pt')
        atomic_json(history,a.output/'history.json');print(json.dumps(history[-1]),flush=True)
        if (a.smoke_steps and steps>=a.smoke_steps) or steps>=a.max_updates:break
    atomic_json(dict(seconds=time.perf_counter()-started,steps=steps,peak_gpu_gib=torch.cuda.max_memory_allocated()/2**30,
        best_sha256=file_sha256(a.output/'best.pt'),last_sha256=file_sha256(a.output/'last.pt'),gradient_proof=gradient_proof,
        all_trainable_parameters_finite_gradients=True,heldout_scored=False),a.output/'freeze.json')


if __name__=='__main__':main()
