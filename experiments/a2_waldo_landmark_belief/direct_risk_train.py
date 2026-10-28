"""Additional full20+5 risk-head training after preserved relation20+5.

Only cached public features enter the head. Original TRAIN/dev remain isolated
from all independent confirmation and benchmark observations.
"""
import argparse,copy,hashlib,json,math,os,subprocess,time
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from .paths import ROOT,A0_SHA,save,sha
from .collected_snapshot import BASE
from .direct_risk_head import DirectRiskHead,head_loss,outcome_classes
from .direct_risk_data import load_original_risk_data,normalization
from .direct_risk_protocol import operating_points,select_point,point_key,GAIN_QUANTILES,HARM_QUANTILES,TEMPERATURES
from .revision_confirmation import immutable_json
from .train import atomic_checkpoint,atomic_copy,write_record_scalars
from .artifact_protocol import verify_curves

DIRECTORY=BASE/'revision_v2/head_training'
CONFIG=dict(version='a2_direct_risk_v2',seed=0,discrimination_epochs=20,refinement_epochs=5,batch_size=512,
    lr=.01,weight_decay=.001,gradient_clip=5.,workers=0,refinement_cost_weight=.2,harm_cost=10.,fixed_action=.25,
    stage21='continue epoch20 weights and optimizer',gain_quantiles=list(GAIN_QUANTILES),harm_quantiles=list(HARM_QUANTILES),
    temperatures=list(TEMPERATURES),selection='dev safe operatingpoint primary; AUC auxiliary; all25 completed first')


def code_hashes():
    names=('direct_risk_train.py','direct_risk_head.py','direct_risk_data.py','direct_risk_protocol.py','risk_probe.py',
           'revision_fusion.py','revision_confirmation.py','train.py','artifact_protocol.py','paths.py')
    return {name:sha(ROOT/'experiments/a2_waldo_landmark_belief'/name) for name in names}


def fingerprint(model):
    h=hashlib.sha256()
    for name,value in model.state_dict().items():h.update(name.encode());h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def prepare():
    data,identities,lineage=load_original_risk_data();stats=normalization(data['train'])
    torch.set_num_threads(2);torch.manual_seed(0)
    model=DirectRiskHead(*stats)
    signature=dict(config=CONFIG,code_sha256=code_hashes(),lineage=lineage,a0_sha256=A0_SHA,
        initial_model_sha256=fingerprint(model),normalization=dict(mean=stats[0].tolist(),std=stats[1].tolist(),class_weights=stats[2].tolist()),
        previous_relation_schedule='completed20ranking+5belief refinement, frozen unchanged',
        new_head_schedule='additional20risk discrimination+5belief risk refinement',benchmark_used=False,confirmation_used=False)
    return data,identities,model,signature


def train_epoch(model,x,y,optimizer,*,epoch,batch_size=512):
    model.train();start=time.perf_counter();views=updates=0;totals=dict(loss=0.,classification=0.,risk_cost=0.)
    order=torch.randperm(len(x),generator=torch.Generator().manual_seed(CONFIG['seed']+epoch))
    for indices in order.split(batch_size):
        indices=indices.to(x.device);logits=model(x[indices]);loss,parts=head_loss(logits,y[indices],model.class_weights,refine=epoch>20)
        if not torch.isfinite(loss):raise ValueError('Nonfinite direct risk loss')
        optimizer.zero_grad(set_to_none=True);loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),CONFIG['gradient_clip'],error_if_nonfinite=True);optimizer.step()
        size=len(indices);views+=size;updates+=1;totals['loss']+=float(loss.detach())*size
        for k in parts:totals[k]+=parts[k]*size
    if views!=len(x):raise ValueError('Incomplete head epoch')
    return dict(**{k:v/views for k,v in totals.items()},views=views,optimizer_updates=updates,seconds=time.perf_counter()-start)


@torch.no_grad()
def evaluate_head(model,data,identities):
    # CPU evaluation makes saved-checkpoint selection exactly reproducible
    # without starting a GPU or depending on its matrix multiplication order.
    model=copy.deepcopy(model).cpu().eval();x=torch.from_numpy(data['features']).float();y=outcome_classes(torch.from_numpy(data['outcomes'][:,-1]))
    logits=model(x);loss,parts=head_loss(logits,y,model.class_weights,refine=False);corrected=model.corrected_logits(logits)
    scores=corrected.softmax(-1).numpy();available=data['features'][:,-1]>0
    points=operating_points(scores,available,identities,data['before_ne'],data['after_ne'][:,-1]);chosen=select_point(points,dev=True)
    active=y!=0;ranking=torch.sigmoid(corrected[:,1]-corrected[:,2]).numpy()
    auc=float(roc_auc_score((y[active]==1).numpy(),ranking[active.numpy()]))
    summary=chosen['summary'] if chosen else {}
    return dict(loss=float(loss),**parts,views=len(x),optimizer_updates=0,gain_harm_auc=auc,selected_point=chosen,
        operating_points=points,eligible=chosen is not None,mean_r1_difference=summary.get('mean_r1_difference',0.),
        corrected=summary.get('corrected',0),broken=summary.get('broken',0),supported_groups=summary.get('supported_groups',0),
        predicted_class_accuracy=float((logits.argmax(-1)==y).float().mean()))


def selection_key(record):
    dev=record['dev'];point=dev['selected_point']
    return (point is not None,*(point_key(point) if point is not None else (0.,)*7),dev['gain_harm_auc'],-dev['classification'],-record['epoch'])


def verify_history(history):
    if len(history)!=25 or [r['epoch'] for r in history]!=list(range(1,26)):raise ValueError('Additional head requires all25 sequential epochs')
    for record in history:
        e=record['epoch'];refine=e>20
        if record['stage']!=('belief_risk_refinement' if refine else 'risk_discrimination') or record['stage_epoch']!=(e-20 if refine else e):raise ValueError('Risk stage schedule changed')
        for split,count in (('train',31415),('dev',4618)):
            m=record[split];steps=math.ceil(count/CONFIG['batch_size']) if split=='train' else 0
            if m['views']!=count or m['optimizer_updates']!=steps or not math.isfinite(m['loss']):raise ValueError('Risk epoch population/update/loss invalid')


def require_idle_gpu():
    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
    if apps:raise ValueError('GPU occupied; do not start concurrent A2 smoke/training')


def smoke():
    require_idle_gpu();data,identities,model,signature=prepare()
    initial=fingerprint(model);model=model.cuda();torch.backends.cuda.matmul.allow_tf32=False
    x=torch.from_numpy(data['train']['features'][:512]).cuda();y=outcome_classes(torch.from_numpy(data['train']['outcomes'][:512,-1])).cuda()
    optimizer=torch.optim.AdamW(model.parameters(),lr=CONFIG['lr'],weight_decay=CONFIG['weight_decay']);losses=[]
    for step in range(6):
        optimizer.zero_grad(set_to_none=True);loss,_=head_loss(model(x),y,model.class_weights,refine=step>=3);loss.backward()
        gradients=[p.grad for p in model.parameters()]
        if not torch.isfinite(loss) or any(g is None or not torch.isfinite(g).all() or not bool(g.norm()>0) for g in gradients):raise ValueError('Risk CUDA backward failed')
        optimizer.step();losses.append(float(loss))
    torch.cuda.synchronize()
    if fingerprint(model)==initial:raise ValueError('CUDA smoke never updated the new head')
    report=dict(status='direct_risk_real_TRAIN_cuda_backward_passed',signature=signature,gpu=torch.cuda.get_device_name(),steps=6,losses=losses,
                initial_model_sha256=initial,foundation_parameters_loaded_for_training=False)
    save(report,BASE/'revision_v2/cuda_smoke.json');print(json.dumps(dict(status=report['status'],losses=losses)),flush=True)


def run_training(*,resume=False):
    require_idle_gpu();data,identities,model,signature=prepare();smoke_path=BASE/'revision_v2/cuda_smoke.json'
    s=json.loads(smoke_path.read_text())
    if s['status']!='direct_risk_real_TRAIN_cuda_backward_passed' or s['signature']!=signature or s['initial_model_sha256']!=fingerprint(model):raise ValueError('Matching real CUDA backward and fresh seed0 initialization required')
    DIRECTORY.mkdir(parents=True,exist_ok=True)
    if (DIRECTORY/'training_complete.json').exists():raise ValueError('Already complete; verify instead of retraining')
    immutable_json(dict(signature,smoke_sha256=sha(smoke_path)),DIRECTORY/'protocol.json');digest=sha(DIRECTORY/'protocol.json')
    model=model.cuda();torch.backends.cuda.matmul.allow_tf32=False
    optimizer=torch.optim.AdamW(model.parameters(),lr=CONFIG['lr'],weight_decay=CONFIG['weight_decay'])
    history=[];first=1
    if resume:
        state=torch.load(DIRECTORY/'last.pt',map_location='cpu',weights_only=False)
        if state['signature_sha256']!=digest:raise ValueError('Resume code/data/source changed')
        e=state['epoch']
        if (e not in range(1,26) or len(state['history'])!=e or [r['epoch'] for r in state['history']]!=list(range(1,e+1))
            or sha(DIRECTORY/'last.pt')!=sha(DIRECTORY/f'epoch_{e:02d}.pt')):raise ValueError('Incomplete/corrupt resume checkpoint')
        if any(not torch.equal(state['model'][key],getattr(model,key).cpu()) for key in ('mean','std','class_weights')):
            raise ValueError('Resume TRAIN statistics changed')
        model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer']);history=state['history'];first=state['epoch']+1
    elif (DIRECTORY/'last.pt').exists():raise ValueError('Existing run requires explicit resume')
    x=torch.from_numpy(data['train']['features']).cuda();y=outcome_classes(torch.from_numpy(data['train']['outcomes'][:,-1])).cuda()
    from torch.utils.tensorboard import SummaryWriter
    writer=SummaryWriter(str(DIRECTORY/'tensorboard'))
    save(history,DIRECTORY/'history.json')
    for record in history:write_record_scalars(record,writer)
    for epoch in range(first,26):
        train=train_epoch(model,x,y,optimizer,epoch=epoch);dev=evaluate_head(model,data['dev'],identities['dev'])
        record=dict(epoch=epoch,stage='belief_risk_refinement' if epoch>20 else 'risk_discrimination',stage_epoch=epoch-20 if epoch>20 else epoch,train=train,dev=dev)
        history.append(record)
        state=dict(signature_sha256=digest,epoch=epoch,model={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
            optimizer=optimizer.state_dict(),history=history,checkpoint_eligible=dev['eligible'],rank_checkpoint_sha256=signature['lineage']['rank_checkpoint_sha256'])
        path=DIRECTORY/f'epoch_{epoch:02d}.pt';atomic_checkpoint(state,path);atomic_copy(path,DIRECTORY/'last.pt');save(history,DIRECTORY/'history.json')
        write_record_scalars(record,writer);writer.flush()
        print(json.dumps(dict(epoch=epoch,stage=record['stage'],train_loss=train['loss'],dev_loss=dev['loss'],dev_auc=dev['gain_harm_auc'],
            eligible=dev['eligible'],dev_gain=dev['corrected'],dev_harm=dev['broken'],dev_group_r1_difference=dev['mean_r1_difference'])),flush=True)
    writer.close();verify_history(history);chosen=max(history,key=selection_key)
    atomic_copy(DIRECTORY/f'epoch_{chosen["epoch"]:02d}.pt',DIRECTORY/'best.pt')
    save(dict(status='direct_risk_full25_complete_confirmation_pending',additional_discrimination_epochs=20,additional_refinement_epochs=5,
        selected_epoch=chosen['epoch'],checkpoint_eligible=chosen['dev']['eligible'],checkpoint_sha256=sha(DIRECTORY/'best.pt'),
        history_sha256=sha(DIRECTORY/'history.json'),signature_sha256=digest),DIRECTORY/'training_complete.json')
    verify_completed_training()


def verify_completed_training():
    data,identities,model,signature=prepare();protocol=json.loads((DIRECTORY/'protocol.json').read_text());marker=json.loads((DIRECTORY/'training_complete.json').read_text())
    if protocol!=dict(signature,smoke_sha256=sha(BASE/'revision_v2/cuda_smoke.json')):raise ValueError('Head protocol/source changed')
    if marker['status']!='direct_risk_full25_complete_confirmation_pending' or marker['additional_discrimination_epochs']!=20 or marker['additional_refinement_epochs']!=5 or marker['signature_sha256']!=sha(DIRECTORY/'protocol.json'):
        raise ValueError('Completed additional20+5 head marker required')
    history=json.loads((DIRECTORY/'history.json').read_text());verify_history(history)
    if marker['history_sha256']!=sha(DIRECTORY/'history.json'):raise ValueError('Head history changed')
    previous=None;updates=0
    for record in history:
        epoch=record['epoch'];state=torch.load(DIRECTORY/f'epoch_{epoch:02d}.pt',map_location='cpu',weights_only=False)
        updates+=record['train']['optimizer_updates']
        if state['epoch']!=epoch or state['history']!=history[:epoch] or state['signature_sha256']!=marker['signature_sha256'] or state['rank_checkpoint_sha256']!=signature['lineage']['rank_checkpoint_sha256']:
            raise ValueError('Completed head checkpoint provenance changed')
        if any(not torch.isfinite(t).all() for t in state['model'].values()):raise ValueError('Nonfinite head checkpoint')
        if previous is not None and all(torch.equal(t,previous[k]) for k,t in state['model'].items()):raise ValueError('No head weights updated in epoch')
        previous=state['model']
        for key in ('mean','std','class_weights'):
            if not torch.equal(state['model'][key],getattr(model,key)):raise ValueError('TRAIN normalization changed while training')
        counter=max((int(v['step']) for v in state['optimizer']['state'].values() if 'step' in v),default=0)
        if counter!=updates:raise ValueError('Head optimizer step counter changed')
        model.load_state_dict(state['model']);actual=evaluate_head(model,data['dev'],identities['dev'])
        if actual!=record['dev'] or state['checkpoint_eligible']!=actual['eligible']:raise ValueError('Saved head dev metrics do not reproduce from actual weights')
    chosen=max(history,key=selection_key);path=DIRECTORY/f'epoch_{chosen["epoch"]:02d}.pt'
    if (sha(path)!=sha(DIRECTORY/'best.pt') or marker['checkpoint_sha256']!=sha(path) or marker['selected_epoch']!=chosen['epoch']
        or marker['checkpoint_eligible']!=chosen['dev']['eligible']):raise ValueError('Head checkpoint violates fixed dev selection')
    curves=verify_curves(history,DIRECTORY)
    report=dict(status='direct_risk_all25_artifacts_verified',checkpoint_sha256=marker['checkpoint_sha256'],selected_epoch=chosen['epoch'],
        checkpoint_eligible=chosen['dev']['eligible'],dev=chosen['dev'],optimizer_steps=updates,all_dev_epochs_recomputed=True,curves=curves,lineage=signature['lineage'])
    save(report,DIRECTORY/'artifact_verification.json');print(json.dumps(dict(status=report['status'],epoch=chosen['epoch'],eligible=report['checkpoint_eligible'])),flush=True)
    return report


def main():
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=('smoke','train','verify'),required=True);p.add_argument('--resume',action='store_true');a=p.parse_args()
    if a.mode=='smoke':smoke()
    elif a.mode=='train':run_training(resume=a.resume)
    else:verify_completed_training()


if __name__=='__main__':main()
