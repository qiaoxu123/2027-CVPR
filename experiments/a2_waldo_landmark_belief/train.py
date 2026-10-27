"""Full seed0 20 ranking +5 refinement epochs, TRAIN-dev checkpoint selection.

Only the new small network is instantiated for optimization. All original
backbone features are immutable, provenance-bound caches. No benchmark split is
read. Smoke/partial caches cannot be used by this entrypoint.
"""
import argparse
import json
from pathlib import Path
import shutil
import time
import torch
from torch.utils.data import Dataset,DataLoader
from .paths import ROOT,A0_SHA,sha,save
from .model import LandmarkBeliefNetwork,supervision_loss
from .batching import collate_views,to_device
from .training_protocol import RANK_EPOCHS,REFINE_EPOCHS,TRAIN_CONFIG,validate_complete_cache,ranking_metrics,selection_key
from .observed_geometry import gaussian_belief_mix


class CachedViews(Dataset):
    def __init__(self,directory,manifest,source,partition):
        self.directory=directory
        self.entries=[e for e in manifest['entries'] if e['split']==partition]
        self.signature=manifest['signature_sha256']
        self.source={e['index']:e for e in source['entries']}
        self.partition=partition

    def __len__(self):return len(self.entries)

    def __getitem__(self,index):
        entry=self.entries[index]
        row=torch.load(self.directory/entry['filename'],map_location='cpu',weights_only=False)
        expected=self.source[entry['episode']]
        if row['signature_sha256']!=self.signature or row['episode']!=entry['episode'] or row['step']!=entry['step']:
            raise ValueError('Cache row provenance/observation identity mismatch')
        if row['target_instance']!=[expected['map_name'],expected['object_id']] or expected['split']!=self.partition:
            raise ValueError('Cache row target identity or partition changed')
        return row


def _loader(dataset,*,epoch,shuffle):
    generator=torch.Generator().manual_seed(TRAIN_CONFIG['seed']+epoch)
    return DataLoader(dataset,batch_size=TRAIN_CONFIG['batch_size'],shuffle=shuffle,
                      num_workers=TRAIN_CONFIG['workers'],collate_fn=collate_views,
                      generator=generator,pin_memory=True)


def run_epoch(model,loader,*,optimizer=None,refine=False,device='cuda'):
    model.train(optimizer is not None)
    totals={key:0. for key in ('loss','ranking','direction','distance','match','offset')}
    reciprocal=[];correct=[];offset_errors=[]
    views=rank_views=updates=0
    probe=dict(a0_correct=0,a2_correct=0,corrected=0,broken=0,unchanged_belief=0)
    started=time.perf_counter()
    with torch.set_grad_enabled(optimizer is not None):
        for number,raw in enumerate(loader,1):
            batch=to_device(raw,device)
            x,y=batch['network'],batch['labels']
            output=model(**x)
            loss,stats=supervision_loss(output,y,refine=refine)
            if not torch.isfinite(loss):raise ValueError('Nonfinite A2 loss')
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                has_supervision=bool((y['angle_valid'] & output['reference_valid']).any() or
                                     (y['rank_label_mask'] & output['reliable']).any())
                if has_supervision:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(),TRAIN_CONFIG['gradient_clip'],error_if_nonfinite=True)
                    optimizer.step();updates+=1
            size=len(x['box_rgb']);views+=size;rank_views+=stats['rank_views']
            totals['loss']+=float(loss.detach())*size
            for key in totals:
                if key!='loss':totals[key]+=stats[key]*size
            rr,top=ranking_metrics(output['scores'].detach(),y['ranking'],y['rank_label_mask'],output['reliable'])
            reciprocal.extend(rr);correct.extend(top)
            valid=y['offset_valid'] & output['reliable']
            if valid.any():
                offset_errors.extend((output['offset'].detach()[valid]-y['offset'][valid]).abs().mean(-1).cpu().tolist())
            if optimizer is None:
                # A fixed internal-dev diagnostic, not an enabled/calibrated
                # runtime gate and not an extra trained ablation arm.
                s=batch['spatial']
                for i in range(size):
                    base=s['base_logits'][i];grid=s['grid_world'][i]
                    eligible=output['reliable'][i] & (output['candidate_match_logits'][i].sigmoid()>=.5)
                    mixed=gaussian_belief_mix(base,grid,s['candidate_xy'][i]+output['offset'][i],
                                             output['scores'][i],s['candidate_support'][i],eligible,.25)
                    before=bool((grid[base.argmax()]-y['target_xy'][i]).norm()<=20)
                    after=bool((grid[mixed.argmax()]-y['target_xy'][i]).norm()<=20)
                    probe['a0_correct']+=int(before);probe['a2_correct']+=int(after)
                    probe['corrected']+=int(after and not before);probe['broken']+=int(before and not after)
                    probe['unchanged_belief']+=int(torch.equal(base,mixed))
            if number%100==0:
                print(json.dumps(dict(batch=number,views=views,rank_views=rank_views,
                                      training=optimizer is not None,loss=totals['loss']/views,
                                      seconds=time.perf_counter()-started)),flush=True)
    if views!=len(loader.dataset):raise ValueError('Epoch did not visit the entire dataset')
    return dict(**{key:value/max(views,1) for key,value in totals.items()},views=views,
                informative_views=rank_views,optimizer_updates=updates,
                known_candidate_mrr=sum(reciprocal)/len(reciprocal) if reciprocal else None,
                known_candidate_top1=sum(correct)/len(correct) if correct else None,
                offset_mae_m=sum(offset_errors)/len(offset_errors) if offset_errors else None,
                internal_dev_probe=probe if optimizer is None else None,
                seconds=time.perf_counter()-started)


def atomic_checkpoint(state,path):
    temp=path.with_suffix('.tmp');torch.save(state,temp);temp.replace(path)


def atomic_copy(source,target):
    temp=target.with_suffix('.tmp');shutil.copyfile(source,temp);temp.replace(target)


def write_record_scalars(record,writer):
    for split,metrics in [('Train',record['train']),('Dev',record['dev'])]:
        for name,value in metrics.items():
            if isinstance(value,(int,float)):
                writer.add_scalar(f'{split}/{name}',value,record['epoch'])
            elif isinstance(value,dict):
                for subkey,subvalue in value.items():
                    if isinstance(subvalue,(int,float)):
                        writer.add_scalar(f'{split}/{name}/{subkey}',subvalue,record['epoch'])


def restore_training_logs(history,directory,writer):
    """Recover interrupted final-epoch logging from authoritative last.pt history."""
    if [h['epoch'] for h in history]!=list(range(1,len(history)+1)):
        raise ValueError('Resume history has missing/repeated epochs')
    save(history,directory/'history.json')
    for record in history:write_record_scalars(record,writer)
    writer.flush()


def reconcile_best_aliases(history,directory):
    """Discard best aliases written ahead of last.pt during an interrupted epoch."""
    keys=[]
    for refine,alias in [(False,'best_rank.pt'),(True,'best.pt')]:
        completed=[h for h in history if (h['epoch']>RANK_EPOCHS)==refine]
        if not completed:
            keys.append(None)
            continue
        chosen=max(completed,key=lambda h:selection_key(h['dev'],refine=refine))
        source=directory/f'epoch_{chosen["epoch"]:02d}.pt'
        state=torch.load(source,map_location='cpu',weights_only=False)
        if state['epoch']!=chosen['epoch'] or state['runtime_code_sha256']!=runtime_hashes() or state['training_modules_sha256']!=training_hashes():
            raise ValueError('Selected epoch checkpoint provenance mismatch during resume')
        atomic_copy(source,directory/alias)
        keys.append(selection_key(chosen['dev'],refine=refine))
    return keys


def runtime_hashes():
    names=['model.py','core.py','observed_geometry.py','mentions.py','data_protocol.py','perception.py','policy.py','paths.py']
    return {name:sha(ROOT/'experiments/a2_waldo_landmark_belief'/name) for name in names}


def training_hashes():
    return {name:sha(ROOT/'experiments/a2_waldo_landmark_belief'/name)
            for name in ('train.py','training_protocol.py','batching.py','model.py','core.py','observed_geometry.py')}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--cache',type=Path,default=ROOT/'runs/a2_waldo_landmark_20261027/cache')
    parser.add_argument('--output',type=Path,default=ROOT/'runs/a2_waldo_landmark_20261027/training')
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    from .prepare import cache_signature
    from torch.utils.tensorboard import SummaryWriter
    saved_signature=json.loads((args.cache/'signature.json').read_text())
    if cache_signature()!=saved_signature:raise ValueError('Current sources/data differ from full extraction signature')
    source=json.loads((ROOT/'runs/a2_waldo_landmark_20261027/source_manifest.json').read_text())
    manifest=validate_complete_cache(args.cache,source,hash_rows=True)
    smoke=json.loads((ROOT/'reports/a2_execution_20261027/informative_cuda_smoke.json').read_text())
    smoke_hashes={name:sha(ROOT/'experiments/a2_waldo_landmark_belief'/name)
                  for name in ('model.py','core.py','observed_geometry.py','batching.py')}
    if (smoke.get('status')!='informative_train_cuda_backward_passed' or smoke.get('device')!='cuda'
            or smoke.get('preflight_code_sha256')!=sha(ROOT/'experiments/a2_waldo_landmark_belief/verify_rows.py')
            or smoke.get('runtime_code_sha256')!=smoke_hashes
            or smoke.get('signature_sha256')!=manifest['signature_sha256']):
        raise ValueError('Updated-model CUDA backward on the FULL TRAIN cache is required before formal training')
    args.output.mkdir(parents=True,exist_ok=True)
    if (args.output/'training_complete.json').exists():
        raise ValueError('Training already completed; verify existing output instead of retraining')
    code_sha=sha(Path(__file__))
    train=CachedViews(args.cache,manifest,source,'train');dev=CachedViews(args.cache,manifest,source,'dev')
    torch.set_num_threads(2);torch.manual_seed(0);torch.cuda.manual_seed_all(0)
    model=LandmarkBeliefNetwork().cuda()
    if set(dict(model.named_children()))!={'language','relation','scorer','offset','match'}:
        raise ValueError('Optimizer must contain only explicitly new A2 modules')
    optimizer=torch.optim.AdamW(model.parameters(),lr=TRAIN_CONFIG['lr'],weight_decay=TRAIN_CONFIG['weight_decay'])
    history=[];best_rank=best_refine=None;start_epoch=1
    last=args.output/'last.pt'
    if last.exists():
        if not args.resume:raise ValueError('Existing training requires explicit --resume')
        previous=torch.load(last,map_location='cpu',weights_only=False)
        if (previous['training_code_sha256']!=code_sha or previous['training_modules_sha256']!=training_hashes()
                or previous['runtime_code_sha256']!=runtime_hashes() or previous['config']!=TRAIN_CONFIG
                or previous['cache_signature_sha256']!=manifest['signature_sha256']):
            raise ValueError('Resume code/cache provenance changed')
        model.load_state_dict(previous['model']);optimizer.load_state_dict(previous['optimizer'])
        history=previous['history'];start_epoch=previous['epoch']+1
        if previous['epoch']!=len(history):raise ValueError('Resume epoch/history length mismatch')
        best_rank,best_refine=reconcile_best_aliases(history,args.output)
    # Rebuild all scalar steps after a purge, including epoch25 if logging was
    # interrupted after its last.pt write. No optimizer work is repeated then.
    writer=SummaryWriter(str(args.output/'tensorboard'),purge_step=1)
    restore_training_logs(history,args.output,writer)
    save(dict(config=TRAIN_CONFIG,model_config=model.config,cache_manifest_sha256=sha(args.cache/'manifest.json'),
              cache_signature_sha256=manifest['signature_sha256'],training_code_sha256=code_sha,
              runtime_code_sha256=runtime_hashes(),training_modules_sha256=training_hashes(),a0_sha256=A0_SHA,
              train_views=len(train),dev_views=len(dev),selection_population='TRAIN-dev only'),args.output/'protocol.json')
    for epoch in range(start_epoch,RANK_EPOCHS+REFINE_EPOCHS+1):
        refine=epoch>RANK_EPOCHS
        if epoch==RANK_EPOCHS+1:
            chosen=torch.load(args.output/'best_rank.pt',map_location='cpu',weights_only=False)
            model.load_state_dict(chosen['model'])
            optimizer=torch.optim.AdamW(model.parameters(),lr=TRAIN_CONFIG['lr'],weight_decay=TRAIN_CONFIG['weight_decay'])
        training=run_epoch(model,_loader(train,epoch=epoch,shuffle=True),optimizer=optimizer,refine=refine)
        validation=run_epoch(model,_loader(dev,epoch=0,shuffle=False),refine=refine)
        record=dict(epoch=epoch,stage='refinement' if refine else 'ranking',
                    stage_epoch=epoch-RANK_EPOCHS if refine else epoch,train=training,dev=validation)
        history.append(record)
        key=selection_key(validation,refine=refine)
        improved=(best_refine is None or key>tuple(best_refine)) if refine else (best_rank is None or key>tuple(best_rank))
        if improved:
            if refine:best_refine=key
            else:best_rank=key
        state=dict(model={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},model_config=model.config,
                   epoch=epoch,stage=record['stage'],history=history,optimizer=optimizer.state_dict(),
                   best_rank_key=best_rank,best_refine_key=best_refine,
                   config=TRAIN_CONFIG,a0_sha256=A0_SHA,cache_signature_sha256=manifest['signature_sha256'],
                   training_code_sha256=code_sha,runtime_code_sha256=runtime_hashes(),training_modules_sha256=training_hashes(),
                   checkpoint_eligible=bool(training['informative_views']>0 and validation['informative_views']>0))
        atomic_checkpoint(state,args.output/f'epoch_{epoch:02d}.pt')
        if improved:atomic_copy(args.output/f'epoch_{epoch:02d}.pt',args.output/('best.pt' if refine else 'best_rank.pt'))
        atomic_checkpoint(state,last)
        save(history,args.output/'history.json')
        write_record_scalars(record,writer)
        writer.flush()
        print(json.dumps(record),flush=True)
    writer.close()
    if len(history)!=25 or [h['epoch'] for h in history]!=list(range(1,26)):
        raise ValueError('Missing epochs: complete20+5 training is required')
    selected=torch.load(args.output/'best.pt',map_location='cpu',weights_only=False)
    save(dict(status='full20plus5_training_complete_calibration_pending',rank_epochs=20,refine_epochs=5,
              chosen_epoch=selected['epoch'],checkpoint_eligible=selected['checkpoint_eligible'],
              checkpoint_sha256=sha(args.output/'best.pt'),history_sha256=sha(args.output/'history.json'),
              train_views_per_epoch=len(train),dev_views_per_epoch=len(dev),calibration_used_for_selection=False,
              benchmark_data_used=False),args.output/'training_complete.json')


if __name__=='__main__':
    main()
