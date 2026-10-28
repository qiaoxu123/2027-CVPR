"""Full20+5 fresh seed0 revision training, no benchmark or confirmation inputs."""
import argparse
import json
from pathlib import Path
import time
import torch
from torch.utils.data import DataLoader
from .paths import ROOT,A0_SHA,save,sha
from .collected_snapshot import BASE,verify_cache_origin,cache_source
from .training_protocol import TRAIN_CONFIG,validate_complete_cache,ranking_metrics
from .train import atomic_checkpoint,atomic_copy,write_record_scalars,restore_training_logs
from .batching import collate_views,to_device
from .revision_model import RevisionNetwork
from .revision_data import RevisionViews
from .revision_loss import revision_loss,LOSS_WEIGHTS

CONFIG=dict(TRAIN_CONFIG,selection='rank: full proposal R1 then NE; refine: map R1 then fewer breaks then NE',
            loss_weights=LOSS_WEIGHTS,offset='exact_zero',maximum_gate=.25,version='a2_revision_v1')
COUNT_FIELDS=('proposal_covered_views','proposal_available_views','proposal_correct','a0_correct','map_correct','corrected','broken',
              'a0_ne_sum','map_ne_sum','proposal_ne_sum')
LOSS_FIELDS=('ranking','direction','distance','match','offset','proposal_localization','category','map_loss','preserve','benefit','expected_ne_risk','gate_mean')


def code_hashes():
    names=('revision_train.py','revision_smoke.py','revision_model.py','revision_fusion.py','revision_data.py','revision_loss.py','revision_audit.py',
           'train.py','training_protocol.py','model.py','core.py','batching.py','data_protocol.py','collected_snapshot.py','paths.py')
    return {n:sha(ROOT/'experiments/a2_waldo_landmark_belief'/n) for n in names}


def revision_selection(metrics,*,refine):
    if refine:return (metrics['map_r1'],-metrics['broken'],-metrics['map_ne'])
    return (metrics['proposal_r1'],-metrics['proposal_ne'])


def loader(dataset,epoch,shuffle):
    return DataLoader(dataset,batch_size=CONFIG['batch_size'],num_workers=CONFIG['workers'],shuffle=shuffle,
                      collate_fn=collate_views,pin_memory=True,generator=torch.Generator().manual_seed(CONFIG['seed']+epoch))


def run_epoch(model,batches,*,optimizer=None,refine=False,device='cuda'):
    model.train(optimizer is not None);started=time.perf_counter();views=updates=contrasts=0
    total_loss=0.;losses={k:0. for k in LOSS_FIELDS};counts={k:0. for k in COUNT_FIELDS};reciprocal=[];top=[]
    full_positive_views=full_winners=0
    with torch.set_grad_enabled(optimizer is not None):
        for number,raw in enumerate(batches,1):
            b=to_device(raw,device);out=model(**b['network']);loss,stats=revision_loss(out,b,refine=refine)
            if not torch.isfinite(loss):raise ValueError('Nonfinite revision loss')
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True);loss.backward()
                torch.nn.utils.clip_grad_norm_(filter(lambda p:p.requires_grad,model.parameters()),CONFIG['gradient_clip'],error_if_nonfinite=True)
                optimizer.step();updates+=1
            size=len(out['scores']);views+=size;total_loss+=float(loss.detach())*size;contrasts+=stats['rank_views']
            for k in losses:losses[k]+=stats[k]*size
            for k in counts:counts[k]+=stats[k]
            rr,tt=ranking_metrics(out['scores'].detach(),b['labels']['ranking'],b['labels']['rank_label_mask'],out['reliable'])
            reciprocal.extend(rr);top.extend(tt)
            for s,y,r in zip(out['scores'].detach(),b['labels']['ranking'],out['reliable']):
                positive=(y>0)&r
                if positive.any():full_positive_views+=1;full_winners+=int(positive[s.argmax()])
            if number%100==0:print(json.dumps(dict(batch=number,views=views,training=optimizer is not None,loss=total_loss/views,seconds=time.perf_counter()-started)),flush=True)
    if views!=len(batches.dataset):raise ValueError('Incomplete revision epoch population')
    return dict(loss=total_loss/max(views,1),**{k:v/max(views,1) for k,v in losses.items()},**counts,
                views=views,informative_views=contrasts,optimizer_updates=updates,
                known_candidate_mrr=sum(reciprocal)/len(reciprocal) if reciprocal else None,
                known_candidate_top1=sum(top)/len(top) if top else None,
                full_pool_verified_target_top1=full_winners/full_positive_views if full_positive_views else None,
                positive_candidate_views=full_positive_views,proposal_r1=counts['proposal_correct']/max(views,1),
                proposal_ne=counts['proposal_ne_sum']/counts['proposal_available_views'] if counts['proposal_available_views'] else 1e6,
                map_r1=counts['map_correct']/max(views,1),a0_r1=counts['a0_correct']/max(views,1),
                map_ne=counts['map_ne_sum']/max(views,1),a0_ne=counts['a0_ne_sum']/max(views,1),
                seconds=time.perf_counter()-started)


def reconcile(history,directory,hashes):
    keys=[]
    for refine,alias in ((False,'best_rank.pt'),(True,'best.pt')):
        records=[r for r in history if (r['epoch']>20)==refine]
        if not records:keys.append(None);continue
        chosen=max(records,key=lambda r:revision_selection(r['dev'],refine=refine))
        path=directory/f'epoch_{chosen["epoch"]:02d}.pt';state=torch.load(path,map_location='cpu',weights_only=False)
        if state['code_sha256']!=hashes or state['epoch']!=chosen['epoch']:raise ValueError('Revision best alias provenance changed')
        atomic_copy(path,directory/alias);keys.append(revision_selection(chosen['dev'],refine=refine))
    return keys


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=BASE/'revision_v1/training');p.add_argument('--resume',action='store_true');a=p.parse_args()
    cache=BASE/'collected_snapshot';verify_cache_origin(cache);source=cache_source(cache)
    manifest=validate_complete_cache(cache,source,hash_rows=True);hashes=code_hashes()
    audit=json.loads((BASE/'revision_data_audit.json').read_text())
    if audit['status']!='all_revision_rows_audited' or audit['cache_signature_sha256']!=manifest['signature_sha256']:
        raise ValueError('Revision all-row audit missing or cache changed')
    smoke=json.loads((BASE/'revision_cuda_smoke.json').read_text())
    if smoke['status']!='revision_real_train_cuda_backward_passed' or smoke['code_sha256']!=hashes or smoke['cache_signature_sha256']!=manifest['signature_sha256']:
        raise ValueError('Current revision code CUDA gate required')
    for name,digest in audit['label_code_sha256'].items():
        if sha(ROOT/'experiments/a2_waldo_landmark_belief'/name)!=digest:raise ValueError('Revision labels changed since audit')
    train=RevisionViews(cache,'train');dev=RevisionViews(cache,'dev')
    if audit['reserve_sha256']!=train.reserve_sha256 or smoke['reserve_sha256']!=train.reserve_sha256:raise ValueError('Confirmation reserve changed')
    torch.set_num_threads(2);torch.manual_seed(0);torch.cuda.manual_seed_all(0)
    model=RevisionNetwork().cuda();optimizer=torch.optim.AdamW(filter(lambda x:x.requires_grad,model.parameters()),lr=CONFIG['lr'],weight_decay=CONFIG['weight_decay'])
    a.output.mkdir(parents=True,exist_ok=True)
    if (a.output/'training_complete.json').exists():raise ValueError('Revision already completed; inspect results instead')
    history=[];start=1;rank_key=refine_key=None
    signature=dict(code_sha256=hashes,config=CONFIG,cache_manifest_sha256=sha(cache/'manifest.json'),cache_signature_sha256=manifest['signature_sha256'],reserve_sha256=train.reserve_sha256,a0_sha256=A0_SHA)
    if (a.output/'last.pt').exists():
        if not a.resume:raise ValueError('Explicit --resume required')
        old=torch.load(a.output/'last.pt',map_location='cpu',weights_only=False)
        if any(old[k]!=v for k,v in signature.items()):raise ValueError('Revision resume provenance changed')
        history=old['history'];start=old['epoch']+1;model.load_state_dict(old['model']);optimizer.load_state_dict(old['optimizer'])
        if len(history)!=old['epoch']:raise ValueError('Revision resume epoch/history mismatch')
        rank_key,refine_key=reconcile(history,a.output,hashes)
    from torch.utils.tensorboard import SummaryWriter
    writer=SummaryWriter(str(a.output/'tensorboard'),purge_step=1);restore_training_logs(history,a.output,writer)
    save(dict(signature,model_config=model.config,train_views=len(train),dev_views=len(dev),confirmation_used=False,benchmark_used=False),a.output/'protocol.json')
    for epoch in range(start,26):
        refine=epoch>20
        if epoch==21:
            best=torch.load(a.output/'best_rank.pt',map_location='cpu',weights_only=False);model.load_state_dict(best['model'])
            optimizer=torch.optim.AdamW(filter(lambda x:x.requires_grad,model.parameters()),lr=CONFIG['lr'],weight_decay=CONFIG['weight_decay'])
        trained=run_epoch(model,loader(train,epoch,True),optimizer=optimizer,refine=refine)
        validation=run_epoch(model,loader(dev,0,False),refine=refine)
        record=dict(epoch=epoch,stage='refinement' if refine else 'ranking',stage_epoch=epoch-20 if refine else epoch,train=trained,dev=validation)
        history.append(record);key=revision_selection(validation,refine=refine)
        previous=refine_key if refine else rank_key;improved=previous is None or key>tuple(previous)
        if improved:
            if refine:refine_key=key
            else:rank_key=key
        state=dict(signature,model={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},model_config=model.config,
                   optimizer=optimizer.state_dict(),epoch=epoch,history=history,
                   checkpoint_eligible=bool(trained['proposal_covered_views']>0 and validation['proposal_covered_views']>0 and validation['corrected']>validation['broken']))
        path=a.output/f'epoch_{epoch:02d}.pt';atomic_checkpoint(state,path)
        if improved:atomic_copy(path,a.output/('best.pt' if refine else 'best_rank.pt'))
        atomic_checkpoint(state,a.output/'last.pt');save(history,a.output/'history.json');write_record_scalars(record,writer);writer.flush()
        print(json.dumps(record),flush=True)
    writer.close()
    if len(history)!=25 or [r['epoch'] for r in history]!=list(range(1,26)):raise ValueError('Missing revision20+5 epochs')
    best=torch.load(a.output/'best.pt',map_location='cpu',weights_only=False)
    save(dict(status='revision_full20plus5_complete_independent_confirmation_pending',rank_epochs=20,refine_epochs=5,
              selected_epoch=best['epoch'],checkpoint_sha256=sha(a.output/'best.pt'),checkpoint_eligible=best['checkpoint_eligible'],
              dev=best['history'][-1]['dev'],history_sha256=sha(a.output/'history.json'),confirmation_used=False,benchmark_used=False),a.output/'training_complete.json')


if __name__=='__main__':main()
